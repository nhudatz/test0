import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.header("Mobile Prediction project")

data = pd.read_csv(r'train.csv',)

if st.checkbox('Show Dataframe'):
  st.write(data)
  
  st.write (data.isna().sum())
  st.write (data.columns)



data = pd.data(np.random.randn(10, 5),
  columns = ('col %d' % i
    for i in range(5)))
data
st.write('This is a line_chart.')
st.line_chart(data)
