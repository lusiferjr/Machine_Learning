import pandas as ps
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score
data=ps.read_csv('train.csv')
print(data)
sns.countplot(x='Survived',data=data)
#plt.show()
sns.countplot(x='Survived',hue='Sex',data=data)
#plt.show()
sns.countplot(x='Survived',hue='Pclass',data=data)
#plt.show()
data['Age'].plot.hist()
#plt.show()
print(data.isna().sum())
sns.heatmap(data.isna(),yticklabels=False,cmap='viridis')
#plt.show()
data=data.drop('Cabin',axis=1)
print(data.head(5))
sns.heatmap(data.isna(),yticklabels=False,cmap='viridis')
#plt.show()
data=data.dropna(subset=['Age'])
sns.heatmap(data.isna(),yticklabels=False,cmap='viridis')
#plt.show()
#print(data.isnull().head())
Sex=ps.get_dummies(data['Sex'],drop_first=True)
print(Sex.head())
emb=ps.get_dummies(data['Embarked'],drop_first=True)
print(emb.head())
pclas=ps.get_dummies(data['Pclass'],drop_first=True)
print(pclas.head())
df=ps.concat([data,Sex,emb,pclas],axis=1)
df=df.drop(['Sex','Embarked','Pclass','Name','Ticket','Fare','Age'],axis=1)
print(df.head())
#making regration model
y=df['Survived']
x=df.drop('Survived',axis=1)
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
# print(x_train)
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=3)
#print(x_train)

print(df)

logmodel=LogisticRegression()
logmodel.fit(x,y)
x_test=ps.read_csv('test.csv')
Sex=ps.get_dummies(x_test['Sex'],drop_first=True)
print(Sex.head())
emb=ps.get_dummies(x_test['Embarked'],drop_first=True)
print(emb.head())
pclas=ps.get_dummies(x_test['Pclass'],drop_first=True)
x_test=ps.concat([x_test,Sex,emb,pclas],axis=1)
x_test=x_test.drop('Cabin',axis=1)
x_test=x_test.drop('Age',axis=1)
x_test=x_test.drop(['Sex','Fare','Embarked','Pclass','Name','Ticket'],axis=1)
print(x_test.isna().sum())
print(df.isna().sum())
pre=logmodel.predict(x_test)
print(pre)
x_test['Survived']=pre
x_text=x_test.drop(['SibSp','Parch','male','Q','S',2,3],axis=1)
ans=ps.DataFrame({})
ans=ps.concat([x_test['PassengerId'],x_test['Survived']],axis=1)
ans.to_csv('pred.csv',index=False)
