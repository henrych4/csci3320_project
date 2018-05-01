# Building regression model to train and predict finish_time based on
# Support Vector Regression Model and Gradient Boosting Regression Tree Model.
# Features: ’actual_weight’, ’declared_horse_weight’, ’draw’, ’win_odds’,
# ’jockey_ave_rank’, ’trainer_ave_rank’, ’recent_ave_rank’, ’race_distance’.
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from math import sqrt
import random

# load training and testing data
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('testing.csv')

# drop meaningless first column
df_train.drop(columns='Unnamed: 0', inplace=True)
df_test.drop(columns='Unnamed: 0', inplace=True)

# Prepare input
def get_input(df, target, standardize=False):
    if standardize :
        selected_features = [
            df['actual_weight'].sub(df_train['actual_weight'].mean()).divide(df_train['actual_weight'].std()),
            df['declared_horse_weight'].sub(df_train['declared_horse_weight'].mean()).divide(df_train['declared_horse_weight'].std()),
            df['draw'].sub(df_train['draw'].mean()).divide(df_train['draw'].std()),
            df['win_odds'].sub(df_train['win_odds'].mean()).divide(df_train['win_odds'].std()),
            df['recent_ave_rank'].sub(df_train['recent_ave_rank'].mean()).divide(df_train['recent_ave_rank'].std()),
            df['jockey_ave_rank'].sub(df_train['jockey_ave_rank'].mean()).divide(df_train['jockey_ave_rank'].std()),
            df['trainer_ave_rank'].sub(df_train['trainer_ave_rank'].mean()).divide(df_train['trainer_ave_rank'].std()),
            df['race_distance'].sub(df_train['race_distance'].mean()).divide(df_train['race_distance'].std()),
        ]
        X = np.stack(selected_features, axis=1)
        y = df[target].map(from_str_to_float).sub(df_train[target].map(from_str_to_float).mean()).divide(df_train[target].map(from_str_to_float).std())
    else :
        selected_features = [
            df['actual_weight'].values,
            df['declared_horse_weight'].values,
            df['draw'].values,
            df['win_odds'].values,
            df['recent_ave_rank'].values,
            df['jockey_ave_rank'].values,
            df['trainer_ave_rank'].values,
            df['race_distance'].values
        ]
        X = np.stack(selected_features, axis=1)
        y = df[target].map(from_str_to_float)
    return X, y

# changing the finishing time from m.s.ms string to float
def from_str_to_float(t):
    (m, s, ms) = t.split('.')
    return int(m) * 60 + int(s) + int(ms) * 0.01

# Top_1 is defined as the percentage/probability when the prediction of top_1 horse (horse with shortest finish_time) for each race is actually the true top_1 horse.
def top_1(method):
    if method == "SVR":
        print("Top_1 of SVR")
        df_test['top_1_pred'] = svr_pred_y
    else:
        print("Top_1 of GBRT")
        df_test['top_1_pred'] = gbrt_pred_y
    top_1_correct = 0
    # for each race, check if the index of the dataframe with minimum top_1_pred equals the index of the dataframe with minimum finish_time
    for race in df_test['race_id'].unique():
        flag=False
        df_race = df_test.loc[df_test['race_id']==race]
        for true_min in np.nditer(df_race['finish_time'].map(from_str_to_float).idxmin()):
            for min_pred in np.nditer((df_race['top_1_pred'].idxmin())):
                if true_min == min_pred:
                    flag = True
        if flag:
            top_1_correct += 1
    print(float(top_1_correct)/len(df_test['race_id'].unique()))

# Top_3 is defined as the percentage/probability when the prediction of top_1 horse for each race is actually within true top_3 horses for each race
def top_3(method):
    if method == "SVR":
        print("Top_3 of SVR")
        df_test['top_1_pred'] = svr_pred_y
    else:
        print("Top_3 of GBRT")
        df_test['top_1_pred'] = gbrt_pred_y
    top_3_correct = 0
    for race in df_test['race_id'].unique():
        flag=False
        df_race = df_test.loc[df_test['race_id']==race]
        for true_3rd_min in np.nditer(df_race['finish_time'].map(from_str_to_float).nsmallest(3).index.values):
            for min_pred in np.nditer((df_race['top_1_pred'].idxmin())):
                if true_3rd_min == min_pred:
                    flag = True
        if flag:
            top_3_correct += 1
    print(top_3_correct/len(df_test['race_id'].unique()))

# Average_rank is defined as the average true rank of top_1 horse based on your prediction over all races.
def average_rank(method):
    if method == "SVR":
        print("Average_rank of SVR")
        df_test['top_1_pred'] = svr_pred_y
    else:
        print("Average_rank of GBRT")
        df_test['top_1_pred'] = gbrt_pred_y
    total_rank = 0
    # for each race, check if the index of the dataframe with minimum top_1_pred equals the index of the dataframe with minimum finish_time
    for race in df_test['race_id'].unique():
        df_race = df_test.loc[df_test['race_id']==race]
        for min_pred in np.nditer((df_race['top_1_pred'].idxmin())):
            fin_pos = df_test.loc[df_test.index == min_pred]
            for rank in fin_pos['finishing_position']:
                total_rank += rank
    print(total_rank/len(df_test['race_id'].unique()))

# Support Vector Regression Model(SVR)
# kernel functions can be linear, poly, rbf and sigmoid.
# precomputed is not suitable as according to stackoverflow, kernel='precomputed' can only be used when passing a (n_samples, n_samples) data matrix that represents pairwise similarities for the samples instead of the (n_samples, n_features) rectangular data matrix.
# ref: https://stackoverflow.com/questions/36306555/scikit-learn-grid-search-with-svm-regression

# SVR with standardized data
print("train SVR with standardized data")
svr_train_X, svr_train_y = get_input(df_train, 'finish_time', True)
''' c = 0.175 epsilon=0.225 best'''
svr_model = SVR(kernel='linear',C=0.2,epsilon=0.2)
svr_model.fit(svr_train_X,svr_train_y)

'''
parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[0.1, 1, 10], 'epsilon':[0.01,0.1,0.5,1,2]}
clf = GridSearchCV(SVR(), parameters, cv=5, refit="neg_mean_squared_error", n_jobs=4, scoring="neg_mean_squared_error" )
clf.fit(svr_train_X, svr_train_y)
print(clf.best_estimator_)
clt_test_X, clt_test_y = get_input(df_test, 'finish_time', True)
clt_pred_y = clf.best_estimator_.predict(clt_test_X)
print(mean_squared_error(clt_test_y, clt_pred_y))
get: 
SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
  kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
0.007172954647468784
'''

# RMSE with normalization of SVR
print("RMSE with normalization of SVR:")
svr_test_X, svr_test_y = get_input(df_test, 'finish_time', True)
svr_pred_y = svr_model.predict(svr_test_X)
print(sqrt(mean_squared_error(svr_test_y, svr_pred_y)))
# Top_1 of SVR
top_1("SVR")
# Top_3 of SVR
top_3("SVR")
# Average_Rank of SVR
average_rank("SVR")

'''
# SVR with non-standardized data
print("train SVR with non-standardized data")
svr_train_X, svr_train_y = get_input(df_train, 'finish_time', False)
svr_model = SVR(kernel='linear',C=0.2,epsilon=0.2)
svr_model.fit(svr_train_X,svr_train_y)
# RMSE without normalization of SVR
print("RMSE without normalization of SVR:")
svr_test_X, svr_test_y = get_input(df_test, 'finish_time', False)
svr_pred_y = svr_model.predict(svr_test_X)
print(sqrt(mean_squared_error(svr_test_y, svr_pred_y)))
# Top_1 of SVR
top_1("SVR")
# Top_3 of SVR
top_3("SVR")
# Average_Rank of SVR
average_rank("SVR")
'''

# Gradient Boosting Regression Tree Model(GBRT)
# loss functions could be one of ls, lad, huber, quantile, select one of them and state your reason in prjreport.pdf.
# Second, learning_rate, n_estimators and max_depth are three critical parameters.

# GBRT with standardized data
print("train GBRT with standardized data")
gbr_train_X, gbr_train_y = get_input(df_train, 'finish_time', True)
'''lr:0.1, ne:300, md:2'''
params = {'n_estimators': 300, 'max_depth': 2, 'learning_rate': 0.25, 'loss': 'quantile'}
gbrt_model = GradientBoostingRegressor(**params)
gbrt_model.fit(gbr_train_X, gbr_train_y)

'''
parameters = {'loss':('ls', 'lad', 'huber', 'quantile'), 'max_depth':[1,2], 'learning_rate':[0.25,0.1,0.5],'n_estimators':[200,300]}
clf = GridSearchCV(GradientBoostingRegressor(), parameters, cv=5, refit="neg_mean_squared_error", n_jobs=4, scoring="neg_mean_squared_error" )
clf.fit(svr_train_X, svr_train_y)
print(clf.best_estimator_)
clt_test_X, clt_test_y = get_input(df_test, 'finish_time', True)
clt_pred_y = clf.best_estimator_.predict(clt_test_X)
print(mean_squared_error(clt_test_y, clt_pred_y))

get 
GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.25, loss='huber', max_depth=2,
             max_features=None, max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=300,
             presort='auto', random_state=None, subsample=1.0, verbose=0,
             warm_start=False)
'''


# RMSE with normalization of GBRT
print("RMSE with normalization of GBRT:")
gbrt_test_X, gbrt_test_y = get_input(df_test, 'finish_time', True)
gbrt_pred_y = gbrt_model.predict(gbrt_test_X)
print(sqrt(mean_squared_error(gbrt_test_y, gbrt_pred_y)))
# Top_1 of GBRT
top_1("GBRT")
# Top_3 of GBRT
top_3("GBRT")
# Average_Rank of GBRT
average_rank("GBRT")

# GBRT with non-standardized data
print("train GBRT with non-standardized data")
gbr_train_X, gbr_train_y = get_input(df_train, 'finish_time', False)
params = {'n_estimators': 300, 'max_depth': 2, 'learning_rate': 0.25, 'loss': 'quantile'}
gbrt_model = GradientBoostingRegressor(**params)
gbrt_model.fit(gbr_train_X, gbr_train_y)
# RMSE without normalization of GBRT
print("RMSE without normalization of GBRT:")
gbrt_test_X, gbrt_test_y = get_input(df_test, 'finish_time', False)
gbrt_pred_y = gbrt_model.predict(gbrt_test_X)
print(sqrt(mean_squared_error(gbrt_test_y, gbrt_pred_y)))
# Top_1 of GBRT
top_1("GBRT")
# Top_3 of GBRT
top_3("GBRT")
# Average_Rank of GBRT
average_rank("GBRT")


'''
# Gridsearch

parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[0.1, 1, 10], 'epsilon':[0.01,0.1,0.5,1,2]}
clf = GridSearchCV(SVR(), parameters, cv=5, refit="neg_mean_squared_error", n_jobs=4, scoring="neg_mean_squared_error" )
clf.fit(svr_train_X, svr_train_y)
print(clf.best_estimator_)
clt_test_X, clt_test_y = get_input(df_test, 'finish_time', True)
clt_pred_y = clf.best_estimator_.predict(clt_test_X)
print(mean_squared_error(clt_test_y, clt_pred_y))

get SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
better.

parameters = {'loss':('ls', 'lad', 'huber', 'quantile'), 'max_depth':[2,3,4,5,6], 'learning_rate':[0.01,0.05,0.1,0.5],'n_estimators':[50,75,100,150,200]}
clf = GridSearchCV(GradientBoostingRegressor(), parameters, cv=5, refit="neg_mean_squared_error", n_jobs=4, scoring="neg_mean_squared_error" )
clf.fit(svr_train_X, svr_train_y)
print(clf.best_estimator_)
clt_test_X, clt_test_y = get_input(df_test, 'finish_time', True)
clt_pred_y = clf.best_estimator_.predict(clt_test_X)
print(mean_squared_error(clt_test_y, clt_pred_y))

get GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='huber', max_depth=5,
             max_features=None, max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=200,
             presort='auto', randomate=None, subsample=1.0, verbose=0,
             warmart=False)
worse.

# 2nd Gridsearch
parameters = { 'C':[0.05, 0.025, 0.75], 'epsilon':[0.75,0.1,0.125]}
clf = GridSearchCV(SVR(kernel='linear'), parameters, cv=5, refit="neg_mean_squared_error", n_jobs=4, scoring="neg_mean_squared_error" )
clf.fit(svr_train_X, svr_train_y)
print(clf.best_estimator_)
clt_test_X, clt_test_y = get_input(df_test, 'finish_time', True)
clt_pred_y = clf.best_estimator_.predict(clt_test_X)
print(mean_squared_error(clt_test_y, clt_pred_y))
worse, end.

parameters = {'loss':('ls', 'lad', 'huber', 'quantile'),'max_depth':[2,3,4,5], 'learning_rate':[0.05,0.075,0.1,0.5],'n_estimators':[50,100,150]}
clf = GridSearchCV(GradientBoostingRegressor(), parameters, refit="neg_mean_squared_error", n_jobs=4, scoring="neg_mean_squared_error" )
clf.fit(svr_train_X, svr_train_y)
print(clf.best_estimator_)
clt_test_X, clt_test_y = get_input(df_test, 'finish_time', True)
clt_pred_y = clf.best_estimator_.predict(clt_test_X)
print(mean_squared_error(clt_test_y, clt_pred_y))
worse,end.
'''
