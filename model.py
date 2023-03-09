# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Simple dataset of three four features 
# First of all load csv dataset file into dataframe
dataset = pd.read_csv(Hiring.csv)
#print the dataset

# feature engineering 
# to replace each Null cell
dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# now specify the Input features for training 
X = dataset.iloc[:, :-1]

# Now some features are string/text, we need to convert them to number
#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.


# train your model using linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print("Test one prediction")
# using predict() function to test your model 
print(model.predict(([7, 9, 6])))