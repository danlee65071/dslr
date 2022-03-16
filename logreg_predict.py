import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def sigmoid(t):
    return 1.0 / (1 + np.exp((-t).astype(float)))


def logit(X, weights):
    return np.dot(X, weights)


def predict_proba(X, weights):
    rows = X.shape[0]
    X_cpy = np.concatenate((np.ones((rows, 1)), X), axis=1)
    return sigmoid(logit(X_cpy, weights))


def predict(X, weights, columns_name, threshold=0.5):
    y_pred = []
    for i in range(weights.shape[0]):
        tmp = predict_proba(X, weights.iloc[i, 1:].to_numpy()) >= threshold
        if len(y_pred) == 0:
            y_pred = tmp
        else:
            y_pred = np.vstack((y_pred, tmp))
    y_pred = pd.DataFrame(y_pred.T, columns=columns_name)
    return y_pred


df_weights = pd.read_csv('weights.csv')
m = df_weights.shape[1]
df_test = pd.read_csv('datasets/dataset_test.csv')
imputer = SimpleImputer(strategy='mean')
df_test_num = df_test.drop(['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)
df_test_fillna = imputer.fit_transform(df_test_num)
df_test = pd.DataFrame(df_test_fillna, columns=df_test_num.columns)
X_test = df_test.drop(['Defense Against the Dark Arts'], axis=1)
scaler = StandardScaler()
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
columns_name = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
y_pred = predict(X_test, df_weights, columns_name)
y_columns = y_pred.columns
for col in y_columns:
    y_pred[col] = np.where(y_pred[col] == 1, col, y_pred[col])
res = np.array([])
for i in range(y_pred.shape[0]):
    res = np.append(res, y_pred.iloc[i].loc[y_pred.iloc[i] != 'False'].index)
res = np.vstack(([range(len(res))], res))
res = pd.DataFrame(res.T, columns=['Index', 'Hogwarts House'])
res.to_csv('houses.csv', index=False)
