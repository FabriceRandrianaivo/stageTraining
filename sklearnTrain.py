import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
# import xlrd

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Model de regression

# # np.random.seed(0)
# # m = 100
# # X = np.linspace(0, 10, m).reshape(m, 1)
# # y= X + np.random.randn(m, 1)

# # plt.scatter(X, y)
# # # plt.show()

# # model = SVR(C=100)#LinearRegression()
# # model.fit(X, y)
# # model.score(X, y)
# # # print(model.predict(X))

# # prediction = model.predict(X)
# # plt.scatter(X,prediction,c= 'r')
# # plt.show()

# # Model Classification--------------------
# data = pd.read_excel('assets/titanic.xls')
# data = data[['survived','pclass','sex','age']]
# data.dropna(axis= 0, inplace=True)

# data['sex'] = data['sex'].replace(['male', 'female'], [0, 1])

# # print(data.head())
# scaler = StandardScaler()
# data[['pclass', 'sex', 'age']] = scaler.fit_transform(data[['pclass', 'sex', 'age']])

# X_train, X_test, y_train, y_test = train_test_split(data[['pclass', 'sex', 'age']], data['survived'], test_size=0.2, random_state=42)

# model = KNeighborsClassifier()

# # y = data['survived']
# # X = data.drop('survived', axis=1)    

# # accuracy = model.score(X_test, y_test)
# # print(f"Précision du modèle : {accuracy}")

# model.fit(X_train, y_train)
# def survie(model, pclass=1, sex=1, age=50):
#     features = scaler.transform([[pclass, sex, age]])
#     prediction = model.predict_proba(features)
#     print(f"Prédiction de survie : {prediction}")

# survie(model)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve 
from sklearn.model_selection import GridSearchCV


iris = load_iris()
X = iris.data
y = iris.target
print(X.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.5, random_state = 5)
print('Train test :',X_train.shape)
print('Test test :',X_test.shape)

model = KNeighborsClassifier()
model.fit(X_train, y_train)
print('Score train:',model.score(X_train, y_train))
print('Score test:',model.score(X_test, y_test))

print(cross_val_score(model, X_train, y_train, cv=5,scoring='accuracy').mean())

k_values = np.arange(1, 50)
train_score, val_score = validation_curve(estimator=model, X=X_train, y=y_train, param_name='n_neighbors', param_range=k_values, cv=5)


plt.plot(k_values,val_score.mean(axis=1), label = 'validation')
plt.plot(k_values,train_score.mean(axis=1), label = 'train')
plt.xlabel('scores')
plt.ylabel('n_neighbors')
plt.legend()
plt.show()



plt.figure(figsize = (12, 4))
plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,alpha = 0.5)
plt.title('Train set')
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,alpha = 0.5)
plt.title('Test set')
# plt.show()
