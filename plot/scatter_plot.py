import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load training and testing data
df_train = pd.read_csv('training.csv')

# drop meaningless first column
df_train.drop(columns='Unnamed: 0', inplace=True)
ax = plt.subplot(1, 2, 1)
ax.set_title("Horse scatter plot")
ax.set_xlabel("win rate")
ax.set_ylabel("number of wins")
horse_list = []
horse_win_num_list = []
horse_win_rate_list = []
for horse in df_train['horse_id'].unique():
    df_horse = df_train.loc[df_train['horse_id']==horse]
    horse_list.append(df_horse['horse_name'].iloc[0])
    number_of_wins = len(df_horse.loc[df_horse['finishing_position']==1])
    horse_win_num_list.append(number_of_wins)
    win_rate = number_of_wins/len(df_horse)
    horse_win_rate_list.append(win_rate)
ax.scatter(horse_win_rate_list, horse_win_num_list, s=10, marker='x')

for i, name in enumerate(horse_list):
    if horse_win_num_list[i] > 6 and horse_win_rate_list[i] > 0.55:
        ax.annotate(name, xy=(horse_win_rate_list[i], horse_win_num_list[i]))

jockey_list = []
jockey_win_num_list = []
jockey_win_rate_list = []
ax = plt.subplot(1, 2, 2)
ax.set_title("Jockey scatter plot")
ax.set_xlabel("win rate")
ax.set_ylabel("number of wins")
for jockey in df_train['jockey'].unique():
    jockey_list.append(jockey)
    df_jockey = df_train.loc[df_train['jockey']==jockey]
    number_of_wins = len(df_jockey.loc[df_jockey['finishing_position']==1])
    jockey_win_num_list.append(number_of_wins)
    win_rate = number_of_wins/len(df_jockey)
    jockey_win_rate_list.append(win_rate)
ax.scatter(jockey_win_rate_list, jockey_win_num_list, s=10, marker='x', c='r')
for i, name in enumerate(jockey_list):
    if jockey_win_num_list[i] > 150 and jockey_win_rate_list[i] > 0.15:
        ax.annotate(name, xy=(jockey_win_rate_list[i], jockey_win_num_list[i]))
plt.show()