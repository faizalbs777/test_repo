import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_csv('Salary_Data.csv')

x=np.array(data['YearsExperience']).reshape(-1,1)

lr=LinearRegression()

lr.fit(x,np.array(data['Salary']))

st.title('Salary Prediction')

nav= st.sidebar.radio("Navigation",["Home","Prediction","contribute"])

if nav =="Home":
    st.image('sal.jpg',width=500)
    if st.checkbox("Show Data"):
        st.table(data)

    graph=st.selectbox("Graph type? ",["Non-Interactive",'Interactive'])

    val= st.slider('Filter data using years of Exp:',0,20)
    data= data.loc[data['YearsExperience']>=val]
    if graph=="Non-Interactive":
        plt.figure(figsize=(10,5))
        plt.scatter(data['YearsExperience'],data['Salary'])
        plt.ylim(0)
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.tight_layout()
        st.pyplot()
    if graph == "Interactive":
        layout = go.layout(
            xaxis= dict(range=[0,16]),
            yaxis= dict(range=[0,210000])
        )
        fig= go.Figure(data=go.scatter(x=data['YearsExperience'],y=data['Salary'],mode='marker'), layout=layout)
        st.plotly_chart(fig)

if nav == "Prediction":

    st.header("What is your salary?")

    val=st.number_input('Enter your experience',0.00,20.00, step=0.25)
    val=np.array(val).reshape(1,-1)
    pred=lr.predict(val)[0]

    if st.button("predict"):
        st.success(f"Your salary is: {round(pred)}")

if nav == "contribute":
    st.write("contrib")