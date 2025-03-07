from email.errors import FirstHeaderLineIsContinuationDefect

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from numpy.conftest import dtype

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
print(df['Species'].value_counts())

#Visualize the species count
sns.pairplot(df,hue='Species', palette='husl')
plt.title("Pair Plot of Iris DataSet")
plt.show()
