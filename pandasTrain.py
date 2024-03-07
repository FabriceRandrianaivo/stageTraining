import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd

data = pd.read_excel('assets/titanic.xls')  
# print(data.head(10))
print(data.shape)
# data = data.drop(['name', 'sibsp', 'ticket', 'cabin', 'embarked', 'boat', 'body', 'home.dest','boat','body','fare','parch'],axis=1)
# print(data.head(10))
# data.fillna(data['age'].mean())
# data = data.dropna(axis= 0)
# print(data.shape)
# print(data.head(10).describe())
# print(data['pclass'].value_counts().plot.bar())
# print(data.groupby(['sex','pclass']).mean())

# data = data.set_index('name')
# print(data['age'].head(10).describe())
# print(data['age'][:10]<10)71
# print(data.loc[:10,['age','sex']])

data.loc[:10,['age','sex']].plot.hist()
plt.show()

# data['age'].plot.hist()
# plt.xlabel('Age')
# plt.ylabel('number')
# plt.show()

# plt.plot(data['age'], data['sex'])
# plt.xlabel('Age')
# plt.ylabel('class')
# plt.title('Relation entre l\'âge et classe')
# plt.show()



