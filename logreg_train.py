import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import copy


# реализация бинарной логистической регрессии
class FtLogisticRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, max_iter=1000, learning_rate=0.1):
        X_cpy = X.copy()
        rows, cols = X_cpy.shape
        self.weights = np.zeros(cols + 1)
        X_cpy = np.concatenate((np.ones((rows, 1)), X_cpy), axis=1)
        for it in range(max_iter):
            p = self._sigmoid(self._logit(X_cpy))
            grad = (1.0 / rows) * np.dot(X_cpy.T, (p - y))
            self.weights -= learning_rate * grad

    def predict_proba(self, X):
        rows = X.shape[0]
        X_cpy = np.concatenate((np.ones((rows, 1)), X), axis=1)
        return self._sigmoid(self._logit(X_cpy))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.weights

    def _sigmoid(self, t):
        return 1.0 / (1 + np.exp(-t))

    def _logit(self, X):
        return np.dot(X, self.weights)


# стохастический градиентный спуск
class FtSGDLogisticRegression(FtLogisticRegression):
    def __init__(self, samples=5):
        super().__init__()
        self.samples = samples

    def _calc_grad(self, X, y, p, rows):
        inds = np.random.choice(np.arange(X.shape[0]), size=self.samples, replace=False)
        return (1 / self.samples) * np.dot(X[inds].T, (p[inds] - y[inds]))


# реализация стратегии OvA для мноногоклассового обучения
class FtOneVsAll:
    def __init__(self, clf):
        self.clf = clf
        self.weights = None
        self.columns_name = None
        self.clfs = []

    def fit(self, X_train, y_train):
        self.columns_name = y_train.columns
        n = y_train.shape[1]
        cols = X_train.shape[1]
        self.weights = np.zeros((n, cols + 1))
        for i in range(n):
            self.clf.fit(X_train, y_train.iloc[:, i])
            self.clfs.append(copy.copy(self.clf))
            self.weights[i] = self.clf.get_weights()

    def predict(self, X):
        y_pred = []
        for clf in self.clfs:
            tmp = clf.predict(X)
            if len(y_pred) == 0:
                y_pred = tmp
            else:
                y_pred = np.vstack((y_pred, tmp))
        y_pred = pd.DataFrame(y_pred.T, columns=self.columns_name)
        return y_pred

    def get_weights(self):
        return self.weights


# загрузка датасета
df_train = pd.read_csv('datasets/dataset_train.csv')
df_train_houses_encoded, df_train_houses_categories = df_train['Hogwarts House'].factorize()
# обработка категориального признака
encoder = OneHotEncoder()
houses_1hot = encoder.fit_transform(df_train_houses_encoded.reshape(-1, 1))
tmp = pd.DataFrame(houses_1hot.toarray(), columns=df_train_houses_categories)
df_train = pd.concat([tmp, df_train.drop(['Hogwarts House', 'Index'], axis=1)], axis=1)
# заполнение пропусков
imputer = SimpleImputer(strategy='mean')
df_train_num = df_train.drop(['First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)
df_train_fillna = imputer.fit_transform(df_train_num)
df_tr = pd.DataFrame(df_train_fillna, columns=df_train_num.columns)
# выбор независимой переменной и ее нормализация
X_train = df_tr.drop(['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff', 'Defense Against the Dark Arts'], axis=1)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# выбор целевой перемены
y_train = df_tr[['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']]
# создание логистической модели
logistic_clf = FtLogisticRegression()
multi_clf = FtOneVsAll(logistic_clf)
# обучение модели
multi_clf.fit(X_train, y_train)
# предсказание модели
y_pred = multi_clf.predict(X_train)
# точность предсказания
accuracy = accuracy_score(y_train, y_pred)
print('accuracy: ', accuracy)
# представление ответа в начальном виде
y_columns = y_train.columns
for col in y_columns:
    y_pred[col] = np.where(y_pred[col] == 1, col, y_pred[col])
res = np.array([])
for i in range(y_pred.shape[0]):
    res = np.append(res, y_pred.iloc[i].loc[y_pred.iloc[i] != 'False'].index)
res = np.vstack(([range(len(res))], res))
res = pd.DataFrame(res.T, columns=['Index', 'Hogwarts House'])
# получение и запись весов в csv
weights = multi_clf.get_weights()
df_weights = pd.DataFrame(weights, index=['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'])
df_weights.to_csv('weights.csv')
# проверка sgd
sgd_clf = FtSGDLogisticRegression()
multi_sgd_clf = FtOneVsAll(sgd_clf)
multi_sgd_clf.fit(X_train, y_train)
y_sgd_pred = multi_sgd_clf.predict(X_train)
print('sgd score: ', accuracy_score(y_train, y_sgd_pred))
