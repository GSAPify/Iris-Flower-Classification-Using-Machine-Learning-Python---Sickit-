from email.errors import FirstHeaderLineIsContinuationDefect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from numpy.conftest import dtype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression()
warnings.simplefilter("ignore")


#Import iris Dataset.
df=pd.read_csv(r'C:\Users\PC\Downloads\iris\iris.data')


# Assign column names to the dataset
df.columns = ['SepalLength', 'SepalWidth', 'PetalLength','PetalWidth', 'Species']

# Dataset Info
df.info()

# Checking for null values
print(df.isnull().sum())

# Display column names
print(df.columns)

#Value counts for Species column
print("Unique Species:", df['Species'].unique())
print("Duplicate Rows:", df.duplicated().sum())

#Visualize the species count
sns.pairplot(df,hue='Species', palette='husl')
sns.set(style="whitegrid") #seaborn style
plt.title("Pair Plot of Iris DataSet")
plt.show()


#Heatmaps
plt.figure(figsize=(10,6))
sns.heatmap(df.drop('Species', axis=1).corr(),annot=True,cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#Define x and y

x=df[['SepalLength', 'SepalWidth','PetalWidth','PetalLength']] #Independent Variables
y=df['Species'] #Dependent Variable

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split (x,y,random_state=0, test_size=0.2) #20% Test Set

#Check Shapes

print("X_train shape:", x_train.shape)
print("X_test shape:", x_test.shape)
print("Y_train shape:", y_train.shape)
print("Y_test shape:", y_test.shape)


model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test,y_pred)*100
conf_matrix = confusion_matrix(y_test,y_pred)


print("Accuracy of the model is {:.2f}%".format(accuracy))
print("Confusion Matrix: ")
print(conf_matrix)

#print Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#visualize confusion report

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_,yticklabels=model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Lables')
plt.title('Confusion Matrix')
plt.show()

print(df.describe())
