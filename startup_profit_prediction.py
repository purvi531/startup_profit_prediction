# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:10:22 2021

@author: HP
"""
#Predicting a Startups Profit/Success Rate using Multiple Linear Regression.
#Here 50 startups dataset containing 5 columns  like “R&D Spend”, “Administration”, “Marketing Spend”, “State”, “Profit”.
#In this dataset first 3 columns provides you spending on Research , Administration and Marketing respectively.
#State indicates startup based on that state. Profit indicates how much profits earned by a startup.

import pandas as pd
df=pd.read_csv("50_Startups.csv");

dummies=pd.get_dummies(df.State) #it will give me dummie values gor State column
#there are 3 unique state names so it create 3 coloumns

merge=pd.concat([df,dummies],axis='columns')  #concating  coloumns in merge var
final=merge.drop(['State'],axis='columns')  #removing state(string type) from dataset


#dropping one of the dummy var for avoiding dummy var trap situation
#if we don't drop it machine will learn wrong model
#here we have 3 dummy var..from 2 we can determine 3rd ..so no need to store extra column
final=final.drop(['New York'],axis='columns') 

#features and lables
x=final.drop(['Profit'],axis='columns') 
y=final['Profit']
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state=1s)
model=LinearRegression() #y=a1x1+a2x2+.....a13x13+b
#fitting data into model
mynewmodel=model.fit(xtrain,ytrain)
y_pred=mynewmodel.predict(xtest)

#don't use it since it is not scaled data(one coloum has 0/1 value while other have 10k range)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(ytest,y_pred) 

from sklearn.metrics import r2_score
score=r2_score(ytest,y_pred)

pred=mynewmodel.predict([[135434,95055,59814,0,0]])