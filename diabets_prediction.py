#importing libraries
import numpy as np 
import pandas as pd  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score ,classification_report ,confusion_matrix  

#Loading the datset
df= pd.read_csv("C:\\Users\\dell\\OneDrive\\Desktop\\diabetes.csv")

#Printing top 5 rows of data
print(df.head())

#Printing datatypes of columns in data
print(df.dtypes)

#Printing to get the summary of data
print(df.info())

#Printing the dimension of data
print(df.shape)

#Printing descriptive statistics of the numerical columns in dataset
print(df.describe().T)

#Printing unique values in column 
print(df['Outcome'].value_counts())

#Grouping the columns with Outcome column and then calculating mean for each
print(df.groupby('Outcome').mean())

#Printiing the correlation between numerical columns
print(df.corr())

#Splitting the dataset 
x=df.drop(columns='Outcome', axis=1)
y=df['Outcome']
print(x)
print(y)

#Standardization of data
scaler=StandardScaler()
scaler.fit(x)
st=scaler.transform(x)
print(st)
x=st

#Splitting data into training and testing sets.
x_train , x_test , y_train , y_test = train_test_split(x ,y , test_size=0.2 , random_state=2) 
print(x.shape , x_train.shape , x_test.shape)

#Training model using Random Forest
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred=rfc.predict(x_test)
print(y_pred)

#calculate the accuracy of a classification model predictions
print(accuracy_score(y_test , y_pred))

#evaluate the performance of a classification model
cm= confusion_matrix(y_test, y_pred)  
print(cm)

#generates a classification report
target_names = ['Diabetic', 'Non-diabetic']
print(classification_report(y_test, y_pred, target_names=target_names))

#using ML model for prediction
input_data=np.array([6,148,75,32,0 ,33.6 ,0.627 , 50])
data_reshape = input_data.reshape(1,-1)
st_input=scaler.transform(data_reshape)
prediction=rfc.predict(st_input)
if(prediction==0):
    print("The person is not diabetic")
else:
    print("Person is diabetic")