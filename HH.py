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

matrix = data.corr()
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(matrix, vmax=1, square=True, annot=True,cmap='Paired')

fig, ax = plt.subplots()
sns.heatmap(df_col.corr(), ax=ax)
st.write(fig)
