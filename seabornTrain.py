import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import seabornTrain as sns
import pandas as pd
from sklearn import datasets

iris = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')

print(iris.head())

sns.pairplot(iris, hue='species')
plt.show()

