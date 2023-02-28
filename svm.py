import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Iris.csv')

# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
print("Shape of dataset : \n",dataset.shape)

print("Top 5 rows of dataset : \n",dataset.head(5))

print("Description of Data : \n",dataset.describe())

# Let’s now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.
print("Each species with total count : \n",dataset.groupby('Species').size())

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = dataset[feature_columns].values
y = dataset['Species'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=0.30)
from sklearn.svm import SVC
model=SVC()

model.fit(x_train, y_train)

pred=model.predict(x_test)

print(confusion_matrix(y_test,pred))

accuracySVM = accuracy_score(y_test, pred)*100
print('Accuracy of our model is equal ' + str(round(accuracySVM, 2)) + ' %.')