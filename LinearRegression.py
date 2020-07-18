import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance

st.write("""
# Simple Score Prediction App
This app predicts the Percentage based on Hours Studied!
""")

st.sidebar.header('User Input Parameter')

def user_input_features():
    hours = st.sidebar.slider('Hours Studied', 0.0, 24.0, 9.25)
    return hours

hours = user_input_features()

st.subheader('User Input parameter')
st.write(hours)

url = "http://bit.ly/w-data"
df = pd.read_csv(url)

#Plot of Distribution of Scores
df.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
st.pyplot()


X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)  

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

st.header('Sample Predictions')
st.subheader(' Hours')
st.write(X_test)
st.subheader('Actual Score')
st.write(y_test)
st.subheader('Predicted Score')
st.write(y_pred)


st.header('Bar Graph Comparing Hours Studied with Score')
df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
st.pyplot()

st.header('Linear Regression Line')
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
st.pyplot()

hours_pred = regressor.predict([[hours]])
st.header('Prediction for User Input')
st.write(hours_pred[0])

st.header('Evaluation Metrics')
st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
