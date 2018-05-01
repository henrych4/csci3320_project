# reference: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# load training and testing data
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('testing.csv')

# drop meaningless first column
df_train.drop(columns='Unnamed: 0', inplace=True)
df_test.drop(columns='Unnamed: 0', inplace=True)

def make_meshgrid(x, y, h=.01):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def get_input(df, target, standardize=False):
    if standardize :
        selected_features = [
            df['recent_ave_rank'].sub(df_train['recent_ave_rank'].mean()).divide(df_train['recent_ave_rank'].std()),
            df['jockey_ave_rank'].sub(df_train['jockey_ave_rank'].mean()).divide(df_train['jockey_ave_rank'].std()),
        ]
        X = np.stack(selected_features, axis=1)
    else :
        selected_features = [
            df['recent_ave_rank'].values,
            df['jockey_ave_rank'].values,
        ]
        X = np.stack(selected_features, axis=1)

    y = []
    for row in df_train.iterrows():
        if int(row[1]['finishing_position']) < len(df_train.loc[df_train['race_id']==str(row[1]['race_id'])]) /2 :
            y.append(1)
        else:
            y.append(0)
    y = np.asarray(y)
    return X, y
svc_train_X, svc_train_y = get_input(df_train, 'finishing_position', False)
svc = SVC(kernel='linear', C=0.1)
X0 = svc_train_X[:, 0]
X1 = svc_train_X[:, 1]
c0 = plt.scatter(X0[np.where(svc_train_y==0)], X1[np.where(svc_train_y==0)], alpha = 0.8, s=5, c='b')
c1 = plt.scatter(X0[np.where(svc_train_y==1)], X1[np.where(svc_train_y==1)], alpha = 0.8, s=5, c='r')
legend = plt.legend([c0, c1], ['not in top 50%', 'in 50%'])

svc.fit(svc_train_X,svc_train_y)
xx, yy = make_meshgrid(X0, X1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.title("visualization of SVM")
plt.xlabel("recent_ave_rank")
plt.ylabel("jockey_ave_rank")
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm )
plt.show()

# svr_pred_y = svr_pred_y.reshape(xx.shape)
