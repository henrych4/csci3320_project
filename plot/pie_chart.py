import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load training and testing data
df_train = pd.read_csv('training.csv')

# drop meaningless first column
df_train.drop(columns='Unnamed: 0', inplace=True)

plt.title("Draw bias effect")
labels = []
fracs = []
for draw in range(1,15):
    labels.append(draw)
    df_draw = df_train.loc[df_train['draw']==draw]
    win_count = len(df_draw[df_draw['finishing_position']==1])
    race_count = len(df_draw)
    fracs.append(win_count/race_count)
    
plt.pie(fracs, labels=labels, autopct='%1.1f%%')
plt.show()