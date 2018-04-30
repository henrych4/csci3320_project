import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load training and testing data
df_train = pd.read_csv('training.csv')

# drop meaningless first column
df_train.drop(columns='Unnamed: 0', inplace=True)

horseID = input("What's horse ID?\n")
if df_train.loc[df_train['horse_id']==str(horseID)].empty :
    print("invalid input!")
else:
    df_horse = df_train.loc[df_train['horse_id']==horseID]
    df_horse = df_horse[-6:]
    plt.title("Recent Racing Result of "+ str(horseID))
    plt.xlabel('race_id')
    plt.ylabel('finishing_position')
    plt.ylim(0, 15)
    plt.plot(df_horse['race_id'],df_horse['finishing_position'])
    plt.show()