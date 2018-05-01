import numpy as np
import pandas as pd

# load horse data
df = pd.read_csv('data/race-result-horse.csv')

# 2.2.1
df = df[df['finishing_position'].apply(lambda x: str(x).isdigit())]

# 2.2.2
horse_rank = {}
horse_new_info = {}
recent_6_runs = []
recent_ave_rank = []
for id in set(df.horse_id):
    horse_rank[id] = []
for i in range(len(df)):
    horse = df.iloc[i]
    horse_rank[horse.horse_id].append(int(horse.finishing_position))
for id in horse_rank:
    recent_6 = horse_rank[id][::-1][:6]
    record = '/'.join([str(x) for x in recent_6])
    if len(recent_6) == 0:
        avg = 7
    else:
        avg = sum(recent_6) / len(recent_6)
    horse_new_info[id] = {'recent_6_runs': record, 'recent_ave_rank': avg}
for id in df['horse_id']:
    recent_6_runs.append(horse_new_info[id]['recent_6_runs'])
    recent_ave_rank.append(horse_new_info[id]['recent_ave_rank'])

df['recent_6_runs'] = recent_6_runs
df['recent_ave_rank'] = recent_ave_rank

# 2.2.3
horse_index = {k: v for v, k in enumerate(list(set(df.horse_id)))}
jockey_index = {k: v for v, k in enumerate(list(set(df.jockey)))}
trainer_index = {k: v for v, k in enumerate(list(set(df.trainer)))}
jockey_record = [[] for i in range(len(jockey_index))]
trainer_record = [[] for i in range(len(trainer_index))]
jockey_ave_rank = []
trainer_ave_rank = []
print('number of horses: {}'.format(len(horse_index)))
print('number of jockeys: {}'.format(len(jockey_index)))
print('number of trainers: {}'.format(len(trainer_index)))
for i in range(len(df)):
    horse = df.iloc[i]
    if horse.race_id > '2016-327':
        splitpoint = i
        break
    jockey_id = jockey_index[horse.jockey]
    trainer_id = trainer_index[horse.trainer]
    jockey_record[jockey_id].append(int(horse.finishing_position))
    trainer_record[trainer_id].append(int(horse.finishing_position))
for i in range(len(df)):
    horse = df.iloc[i]
    jockey_id = jockey_index[horse.jockey]
    trainer_id = trainer_index[horse.trainer]
    if len(jockey_record[jockey_id]) == 0:
        jockey_avg = 7
    else:
        jockey_avg = sum(jockey_record[jockey_id]) / len(jockey_record[jockey_id])
    if len(trainer_record[trainer_id]) == 0:
        trainer_avg = 7
    else:
        trainer_avg = sum(trainer_record[trainer_id]) / len(trainer_record[trainer_id])
    jockey_ave_rank.append(jockey_avg)
    trainer_ave_rank.append(trainer_avg)

df['jockey_ave_rank'] = jockey_ave_rank
df['trainer_ave_rank'] = trainer_ave_rank

# 2.2.4
race_distance = []
df2 = pd.read_csv('data/race-result-race.csv')
race_id_distance = {k: v for k, v in zip(df2['race_id'], df2['race_distance'])}
for i in range(len(df)):
    horse = df.iloc[i]
    race_distance.append(race_id_distance[horse.race_id])

df['race_distance'] = race_distance

# 2.2.5
df_train = df.iloc[:splitpoint]
df_test = df.iloc[splitpoint:]
df_train.to_csv('training.csv')
df_test.to_csv('testing.csv')