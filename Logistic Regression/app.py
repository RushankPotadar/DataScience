#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd


# In[2]:


# Load the trained model
model_file = 'logistic_regression.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)


# In[3]:


# Title
st.title("Titanic Survival Prediction App")


# In[4]:


# User input fields
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 100, 25)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.sidebar.selectbox("Embarked Port", ["Q", "S", "C"])


# In[5]:


# Encode categorical inputs
sex_encoded = 1 if sex == "Male" else 0
embarked_mapping = {"Q": 0, "S": 1, "C": 2}
embarked_encoded = embarked_mapping[embarked]


# In[6]:


# Prepare input for the model
df = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])


# In[7]:


# Predict
if st.sidebar.button("Predict"):
    pred_prob = model.predict_proba(df)

    # Display survival probability
    st.subheader("Survival Probability")
    st.write("Yes" if pred_prob[0][1] >= 0.5 else "No")

    # Show predicted probabilities
    st.subheader("Predicted Probability")
    st.write(pred_prob)

