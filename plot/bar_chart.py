# reference: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# load training and testing data
df_train = pd.read_csv('training.csv')

# drop meaningless first column
df_train.drop(columns='Unnamed: 0', inplace=True)

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
    y = df[target].map(from_float_to_int)
    return X, y

# changing the finishing time from m.s.ms string to float 
def from_float_to_int(t):
    return int(t)

def from_index_to_str(t):
    name = ["actual_weight","declared_horse_weight","draw","win_odds","recent_ave_rank","jockey_ave_rank","trainer_ave_rank","race_distance"]
    return name[t]

# Build a forest and compute the feature importances
rf_train_X, rf_train_y = get_input(df_train, 'finishing_position', True)
# NOT TUNED
rf = RandomForestClassifier(n_estimators=250)

rf.fit(rf_train_X, rf_train_y)
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
names = np.array([from_index_to_str(x) for x in indices])

# Print the feature ranking
print("Feature ranking:")
for f in range(rf_train_X.shape[1]):
    print("%d. %s (%f)" % (f + 1, names[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(rf_train_X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(rf_train_X.shape[1]), names)
plt.xlim([-1, rf_train_X.shape[1]])
plt.show()