# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries
2. Load the dataset using pd.read_csv()
3. Convert the dataset into a dataframe and do preprocessing if required using LabelEncoder
4. Define the input and target variable
5. Split the dataset into training and testing data
6. Train the model using DecisionTreeClassifier(),.fit() and predict using .predict()
7. Measure the accuracy of the model using accuracy_score()
8. Test the model with new input data

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
*/
      import pandas as pd
      data=pd.read_csv("Employee (1).csv")
      data.head()
      data.info()
      
      from sklearn.preprocessing import LabelEncoder
      le=LabelEncoder()
      data["salary"]=le.fit_transform(data['salary'])
      data.head()
      
      x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
      y=data['left']
      x.head()
      y.head()
      
      from sklearn.model_selection import train_test_split
      x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
      from sklearn.tree import DecisionTreeClassifier
      dt=DecisionTreeClassifier(criterion='entropy')
      dt.fit(x_train,y_train)
      y_pred=dt.predict(x_test)
      from sklearn.metrics import accuracy_score
      acc=accuracy_score(y_test,y_pred)
      print(acc)
      dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/user-attachments/assets/b1d9893f-a619-4f39-97b8-5329093bd7b0)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
