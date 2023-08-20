import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import math

# import tensorflow_decision_forests as tfdf

# get the data
cur_path = os.getcwd()
cur_path = cur_path + './titanic'

train_df = pd.read_csv(cur_path  + './train.csv' , index_col = "PassengerId")
test_df = pd.read_csv(cur_path + './test.csv' , index_col = "PassengerId")

# print(test_df.head())
# print(train_df.describe())

###################### preprocess the data #######################

# turing sex into numbers
test_df['Sex'] = test_df['Sex'].replace({'male' : 1 , 'female' : 0})
train_df['Sex'] = train_df['Sex'].replace({'male' : 1 , 'female' : 0})


# combine parch and sibsp
test_sibsp  = test_df['SibSp'].to_numpy()
test_parch = test_df['SibSp'].to_numpy()
test_cnt = test_sibsp + test_parch

train_sibsp  = train_df['SibSp'].to_numpy()
train_parch = train_df['SibSp'].to_numpy()
train_cnt = train_sibsp + train_parch

test_df['cnt'] = test_cnt
train_df['cnt'] = train_cnt

# store the end of Titanic
answer_df = train_df['Survived'].to_numpy()

# drop unnessesary data 
to_drop = ['Name' , 'Ticket' , 'Cabin' , 'Embarked' , 'SibSp' , 'Parch']
test_df.drop(to_drop , inplace = True , axis = 1)
train_df.drop(to_drop , inplace = True , axis = 1)
train_df.drop(['Survived'] , inplace = True , axis = 1)

# fill missing data with median
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

for column in train_df.columns :
    train_df[column] = round((train_df[column] - train_df[column].mean()) / train_df[column].std() , 2) 

for column in test_df.columns :
    test_df[column] = round((test_df[column] - test_df[column].mean()) / test_df[column].std() , 2) 

# feed the model
class log_reg :

    def __init__(self):
        pass

    def training (self , input_df , answer_df , repeat_time , learning_rate):
        para = np.array([1.0,1.0,1.0,1.0,1.0])
        for i in range(0 , repeat_time):
            gradient = self.one_try(input_df , para , answer_df)
            # print(gradient)
            for index in range(0 , 5):
                para[index] -= gradient[index] * learning_rate
            # print(para)
            # print('-----------------------------')      
        return para


    def one_try (self , input_df , para , answer_df):
        cost = 0
        arr = input_df.dot(para)
        gradient = []
        for j , par in enumerate(para):            
            gradient.append(0)
            for i , ele in enumerate(arr):
                p_i = 1 / math.exp(-ele)
                y_i = answer_df[i]
                gradient[j] -= ((y_i - p_i) * (input_df.iat[i , j]))
            gradient[j] /= len(arr)        
        return gradient


test = log_reg()
parameter = test.training(train_df , answer_df , 200 , 0.01)
print(test_df.info())
print(parameter)

predict = []
for i in range(0 , 418):
    p = 0.0
    for j in range(0 , 5):
        p += test_df.iat[i , j] * parameter[j]
    if p > 0.5 :
        predict.append([i+892,1])
    else :
        predict.append([i+892,0])



print(predict)

predict_df = pd.DataFrame(predict , columns = ['PassengerId' , 'Survived'])
predict_df.to_csv(cur_path + './ans.csv' , index = False)

#predict_check = pd.read_csv(cur_path + './gender_submission.csv' , index_col = "PassengerId").to_numpy()
#count = 0
#for i in range(0 , len(predict)):
#    if predict[i] == predict_check[i]:
#        count += 1
#    else :
#        pass









