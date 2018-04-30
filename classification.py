import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


# load training and testing data
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('testing.csv')

# drop meaningless first column
df_train.drop(columns='Unnamed: 0', inplace=True)
df_test.drop(columns='Unnamed: 0', inplace=True)

# preprocess X, y
def get_input(df, target, standardize=True):
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
        y = df[target]
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
        y = df[target]
    return X, y

# print performance
def printPerformance(predict_item, set_type, y_true, y_predict):
    print('Performance of predicting: {} on {} set:'.format(predict_item, set_type))
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
    print('TN: {} | FP: {}'.format(tn, fp))
    print('FN: {} | TP: {}'.format(fn, tp))
    print(classification_report(y_true, y_predict))

def printTimeSpent(start_time, end_time):
    minute, second = divmod(end_time - start_time, 60)
    print('Time spent:{0}m {1}s'.format(int(minute), round(second, 4)))

def exportCsv(name, df_predict, win, top3, top50p):
    name = './predictions/' + name + '_predictions.csv'
    df_predict['HorseWin'] = win
    df_predict['HorseRankTop3'] = top3
    df_predict['HorseRankTop50Percent'] = top50p
    df_predict.to_csv(path_or_buf=name, index=False)

# calculate ground truth
def getTrueLabel(df):
    horseWin = []
    horseRankTop3 = []
    horseRankTop50Percent = []
    start = 0
    while start != len(df):
        end = start
        while end < len(df) and df['race_id'][start] == df['race_id'][end]:
            end += 1
        for i in range(start, end):
            horse = df.iloc[i]
            if horse['finishing_position'] == 1:
                horseWin.append(1)
            else:
                horseWin.append(0)
            if horse['finishing_position'] <= 3:
                horseRankTop3.append(1)
            else:
                horseRankTop3.append(0)
            if horse['finishing_position'] < (end-start)/2:
                horseRankTop50Percent.append(1)
            else:
                horseRankTop50Percent.append(0)
        start = end
    return horseWin, horseRankTop3, horseRankTop50Percent

horseWin, horseRankTop3, horseRankTop50Percent = getTrueLabel(df_train)
df_train['HorseWin'] = horseWin
df_train['HorseRankTop3'] = horseRankTop3
df_train['HorseRankTop50Percent'] = horseRankTop50Percent
horseWin, horseRankTop3, horseRankTop50Percent = getTrueLabel(df_test)
df_test['HorseWin'] = horseWin
df_test['HorseRankTop3'] = horseRankTop3
df_test['HorseRankTop50Percent'] = horseRankTop50Percent
df_predict = df_test[['race_id', 'horse_id']]
df_predict.rename(columns={'race_id': 'RaceId', 'horse_id': 'HorseID'}, inplace=True)

# logistic regression CV
print("=====Logistic Regression=====")
lr_model = LogisticRegressionCV(cv=10, class_weight='balanced', refit=False)

# HorseWin
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseWin')
lr_model.fit(X_train, y_train)
y_predict_train = lr_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseWin')
y_predict_test_win = lr_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseWin', 'Train', y_train, y_predict_train)
printPerformance('HorseWin', 'Test', y_test, y_predict_test_win)
# HorseRankTop3
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseRankTop3')
lr_model.fit(X_train, y_train)
y_predict_train = lr_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseRankTop3')
y_predict_test_top3 = lr_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseRankTop3', 'Train', y_train, y_predict_train)
printPerformance('HorseRankTop3', 'Test', y_test, y_predict_test_top3)
# HorseRankTop50Percent
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseRankTop50Percent')
lr_model.fit(X_train, y_train)
y_predict_train = lr_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseRankTop50Percent')
y_predict_test_half = lr_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseRankTop50Percent', 'Train', y_train, y_predict_train)
printPerformance('HorseRankTop50Percent', 'Test', y_test, y_predict_test_half)
exportCsv('lr', df_predict, y_predict_test_win, y_predict_test_top3, y_predict_test_half)

# Naïve Bayes
# can only use Gaussian Naïve Bayes
print("========Naïve Bayes========")
gnb_model = GaussianNB()

# HorseWin
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseWin')
gnb_model.fit(X_train, y_train)
y_predict_train = gnb_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseWin')
y_predict_test_win = gnb_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseWin', 'Train', y_train, y_predict_train)
printPerformance('HorseWin', 'Test', y_test, y_predict_test_win)
# HorseRankTop3
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseRankTop3')
gnb_model.fit(X_train, y_train)
y_predict_train = gnb_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseRankTop3')
y_predict_test_top3 = gnb_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseRankTop3', 'Train', y_train, y_predict_train)
printPerformance('HorseRankTop3', 'Test', y_test, y_predict_test_top3)
# HorseRankTop50Percent
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseRankTop50Percent')
gnb_model.fit(X_train, y_train)
y_predict_train = gnb_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseRankTop50Percent')
y_predict_test_half = gnb_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseRankTop50Percent', 'Train', y_train, y_predict_train)
printPerformance('HorseRankTop50Percent', 'Test', y_test, y_predict_test_half)
exportCsv('nb', df_predict, y_predict_test_win, y_predict_test_top3, y_predict_test_half)

# SVM
# kernel functions can be linear, poly, rbf and sigmoid.
# precomputed is not suitable as according to stackoverflow, kernel='precomputed' can only be used when passing a (n_samples, n_samples) data matrix that represents pairwise similarities for the samples instead of the (n_samples, n_features) rectangular data matrix.
# ref: https://stackoverflow.com/questions/36306555/scikit-learn-grid-search-with-svm-regression
print("============SVM============")
svm_model = SVC(class_weight='balanced')
param_grid = {
    'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
}
CV_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=4,scoring='precision')

# HorseWin
'''
gridsearch get
SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
'''
svm_model = SVC(kernel='poly',class_weight='balanced')
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseWin')
svm_model.fit(X_train, y_train)
'''
CV_svm.fit(X_train, y_train)
print(CV_svm.best_estimator_)
'''
y_predict_train = svm_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseWin')
y_predict_test_win = svm_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseWin', 'Train', y_train, y_predict_train)
printPerformance('HorseWin', 'Test', y_test, y_predict_test_win)

# HorseRankTop3
'''
gridsearch get
SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
'''
svm_model = SVC(kernel='rbf',class_weight='balanced')
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseRankTop3')
svm_model.fit(X_train, y_train)
'''
CV_svm.fit(X_train, y_train)
print(CV_svm.best_estimator_)
'''
y_predict_train = svm_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseRankTop3')
y_predict_test_top3 = svm_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseRankTop3', 'Train', y_train, y_predict_train)
printPerformance('HorseRankTop3', 'Test', y_test, y_predict_test_top3)

# HorseRankTop50Percent
'''
gridsearch get
SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
'''
# original: rbf
svm_model = SVC(kernel='rbf',class_weight='balanced')
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseRankTop50Percent')
svm_model.fit(X_train, y_train)
'''
CV_svm.fit(X_train, y_train)
print(CV_svm.best_estimator_)
'''
y_predict_train = svm_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseRankTop50Percent')
y_predict_test_half = svm_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseRankTop50Percent', 'Train', y_train, y_predict_train)
printPerformance('HorseRankTop50Percent', 'Test', y_test, y_predict_test_half)
exportCsv('svm', df_predict, y_predict_test_win, y_predict_test_top3, y_predict_test_half)

# Random Forest
print("=======Random Forest=======")
rf_model = RandomForestClassifier(class_weight='balanced')
param_grid = {
    'n_estimators': [50,100,150],
    'max_features': ['auto', 'log2']
}
CV_rfc = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=10, n_jobs=4,scoring='precision')
# HorseWin
'''
gridsearch get
RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)
worse
'''
rf_model = RandomForestClassifier(n_estimators=100,class_weight='balanced')
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseWin')
rf_model.fit(X_train, y_train)
'''
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_estimator_)
'''
y_predict_train = rf_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseWin')
y_predict_test_win = rf_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseWin', 'Train', y_train, y_predict_train)
printPerformance('HorseWin', 'Test', y_test, y_predict_test_win)
# HorseRankTop3
'''
gridsearch get
RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=50, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)
'''
rf_model = RandomForestClassifier(n_estimators=50,class_weight='balanced')
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseRankTop3')
rf_model.fit(X_train, y_train)
'''
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_estimator_)
'''
y_predict_train = rf_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseRankTop3')
y_predict_test_top3 = rf_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseRankTop3', 'Train', y_train, y_predict_train)
printPerformance('HorseRankTop3', 'Test', y_test, y_predict_test_top3)
# HorseRankTop50Percent
'''
gridsearch get
RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)
'''
rf_model = RandomForestClassifier(n_estimators=100,class_weight='balanced')
start_time = time.time()
X_train, y_train = get_input(df_train, 'HorseRankTop50Percent')
rf_model.fit(X_train, y_train)
'''
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_estimator_)
'''
y_predict_train = rf_model.predict(X_train)
X_test, y_test = get_input(df_test, 'HorseRankTop50Percent')
y_predict_test_half = rf_model.predict(X_test)
end_time = time.time()
printTimeSpent(start_time, end_time)
printPerformance('HorseRankTop50Percent', 'Train', y_train, y_predict_train)
printPerformance('HorseRankTop50Percent', 'Test', y_test, y_predict_test_half)
exportCsv('rf', df_predict, y_predict_test_win, y_predict_test_top3, y_predict_test_half)
