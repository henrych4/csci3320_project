import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB

# load training and testing data
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('testing.csv')

# drop meaningless first column
df_train.drop(columns='Unnamed: 0', inplace=True)
df_test.drop(columns='Unnamed: 0', inplace=True)

# preprocess X, y
def get_input_classification(df, target, standardize=True):
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

def get_input_regression(df, target, standardize=False):
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

def betting_classification(model, df_test):
    money = 0
    start = 0
    X_test, _ = get_input_classification(df_test, 'HorseWin')
    while start != len(df_test):
        end = start
        while end < len(df_test) and df_test['race_id'][start] == df_test['race_id'][end]:
            end += 1
        predict_proba = model.predict_proba(X_test[start:end])
        final_predict = np.argmax(predict_proba[:,1]-predict_proba[:,0])
        if df_test['finishing_position'][final_predict] == 1:
            money += df_test['win_odds'][final_predict]
        else:
            money -= 1
        '''
        print money after each race
        print('money after raceID {} = {}'.format(df_test['race_id'][start], money))
        '''
        start = end
    return money

def betting_regression(model, df_test):
    money = 0
    start = 0
    X_test, _ = get_input_regression(df_test, 'finish_time', True)
    while start != len(df_test):
        end = start
        while end < len(df_test) and df_test['race_id'][start] == df_test['race_id'][end]:
            end += 1
        predict_y = model.predict(X_test[start:end])
        final_predict = np.argmin(predict_y)
        if df_test['finishing_position'][final_predict] == 1:
            money += df_test['win_odds'][final_predict]
        else:
            money -= 1
        '''
        print money after each race
        print('money after raceID {} = {}'.format(df_test['race_id'][start], money))
        '''
        start = end
    return money

def ourBetting(model_c1, model_r1, model_r2, df_test):
    money = 0
    start = 0
    X_test_c, _ = get_input_classification(df_test, 'HorseWin')
    X_test_r, _ = get_input_regression(df_test, 'finish_time', True)
    while start != len(df_test):
        end = start
        while end < len(df_test) and df_test['race_id'][start] == df_test['race_id'][end]:
            end += 1
        predict_proba = model_c1.predict_proba(X_test_c[start:end])
        predict_c1 = np.argmax(predict_proba[:,1]-predict_proba[:,0])

        predict_time = model_r1.predict(X_test_r[start:end])
        predict_r1 = np.argmin(predict_time)
        predict_time = model_r2.predict(X_test_r[start:end])
        predict_r2 = np.argmin(predict_time)

        predict_list = [predict_c1, predict_r1, predict_r2]
        predict_final = max(set(predict_list), key=predict_list.count)
        if predict_list.count(predict_final) > 1:
            if df_test['finishing_position'][predict_final] == 1:
                money += df_test['win_odds'][predict_final]
            else:
                money -= 1
            '''
            print money after each race
            print('money after raceID {} = {}'.format(df_test['race_id'][start], money))
            '''
        start = end
    return money

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


print("=====Logistic Regression=====")
lr_model = LogisticRegressionCV(cv=10, class_weight='balanced', refit=False)
X_train, y_train = get_input_classification(df_train, 'HorseWin')
lr_model.fit(X_train, y_train)
money_gained = betting_classification(lr_model, df_test)
print('betting using Logistic Regression: ${}'.format(money_gained))

print("========Naïve Bayes========")
gnb_model = GaussianNB()
X_train, y_train = get_input_classification(df_train, 'HorseWin')
gnb_model.fit(X_train, y_train)
money_gained = betting_classification(gnb_model, df_test)
print('betting using Naïve Bayes: ${}'.format(money_gained))

print("============SVM============")
svm_model = SVC(kernel='poly', class_weight='balanced', probability=True)
X_train, y_train = get_input_classification(df_train, 'HorseWin')
svm_model.fit(X_train, y_train)
money_gained = betting_classification(svm_model, df_test)
print('betting using SVM: ${}'.format(money_gained))

print("=======Random Forest=======")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
X_train, y_train = get_input_classification(df_train, 'HorseWin')
rf_model.fit(X_train, y_train)
money_gained = betting_classification(rf_model, df_test)
print('betting using Random Forest: ${}'.format(money_gained))

print("============SVR============")
svr_train_X, svr_train_y = get_input_regression(df_train, 'finish_time', True)
svr_model = SVR(kernel='linear',C=0.2,epsilon=0.2)
svr_model.fit(svr_train_X,svr_train_y)
money_gained = betting_regression(svr_model, df_test)
print('betting using SVR: ${}'.format(money_gained))

print("============GBRT============")
gbr_train_X, gbr_train_y = get_input_regression(df_train, 'finish_time', True)
params = {'n_estimators': 300, 'max_depth': 2, 'learning_rate': 0.25, 'loss': 'quantile'}
gbrt_model = GradientBoostingRegressor(**params)
gbrt_model.fit(gbr_train_X, gbr_train_y)
money_gained = betting_regression(gbrt_model, df_test)
print('betting using GBRT: ${}'.format(money_gained))

money_gained = ourBetting(svm_model, svr_model, gbrt_model, df_test)
print('betting using our method: ${}'.format(money_gained))
