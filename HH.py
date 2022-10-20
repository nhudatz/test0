import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.header("Mobile Prediction project")

data = pd.read_csv(r'train.csv',)

if st.checkbox('Show Dataframe'):
  st.write(data)
  
  st.write('This is a column.')
  st.write (data.columns)

st.write('This is a pie chart for price range.')

pie_chart = px.pie(data,"price_range")
st.plotly_chart(pie_chart)

#st.write('This is a outlier for px_height.')
#fig,ax = plt.subplots()
#ax.box(data['px_height'],bins=20)
#st.pyplot(fig)

st.write('To see correlation')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

dcopy=data.copy()

dcopy_new=dcopy

dcopy_new[['clock_speed', 'm_dep','fc','px_height']] = dcopy[['clock_speed', 'm_dep','fc','px_height']].astype('int64') 

matrix = dcopy.corr()
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(matrix, vmax=1, square=True, annot=True,cmap='Paired')

fig, ax = plt.subplots()
sns.heatmap(matrix, ax=ax)
st.pyplot(fig)



X=dcopy.drop(['price_range'],axis=1)
y=dcopy[['price_range']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
st.write (dtree.score(X_test,y_test))


#acc_score=accuracy_score(y_test,y_pred)
#st.write('The Accuracy of Model is : ',acc_score)
recall=recall_score(y_test,y_pred,average='weighted')
st.write('The Recall Score of Model is : ',recall)
fscore=f1_score(y_test,y_pred,average='weighted')
st.write('The F-Score of Model is : ',fscore)



#fig, ax = plt.subplots()

#plt.figure(figsize=(10,5))
#sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,annot_kws={'size':10},fmt='d')
#plt.xlabel('Predicted Values',fontsize=14)
#plt.ylabel('Actual Values',fontsize=14)
#plt.title('Confusion Matrix with Acccuracy {}'.format(acc_score),fontsize=16)
#plt.show()

#st.write(fig)
