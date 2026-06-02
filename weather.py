
# WEATHER PATTERN ANALYSIS DASHBOARD 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Weather Pattern Analysis",
    layout="centered"
)

st.title(" WEATHER PATTERN ANALYSIS USING MACHINE LEARNING")

st.write("Analyze weather patterns and predict temperature")


# LOAD DATASET

df = pd.read_csv("weatherHistory.csv")


# DATA CLEANING


# Remove missing values
df.dropna(inplace=True)

# =========================================================
# FIX DATE COLUMN
# =========================================================

# Convert to string
df['Formatted Date'] = df['Formatted Date'].astype(str)

# Remove timezone if present
df['Formatted Date'] = df['Formatted Date'].str.split('+').str[0]

# Convert to datetime
df['Formatted Date'] = pd.to_datetime(
    df['Formatted Date'],
    errors='coerce'
)

# Remove invalid rows
df = df.dropna(subset=['Formatted Date'])


# EXTRACT MONTH AND YEAR


df['Month'] = df['Formatted Date'].dt.month

df['Year'] = df['Formatted Date'].dt.year


# RENAME COLUMNS

df.rename(columns={
    'Temperature (C)': 'Temperature',
    'Humidity': 'Humidity',
    'Wind Speed (km/h)': 'WindSpeed',
    'Pressure (millibars)': 'Pressure'
}, inplace=True)


# DATASET PREVIEW

st.subheader(" Dataset Preview")

st.dataframe(df.head())

# =========================================================
# DATASET INFORMATION
# =========================================================

st.subheader(" Dataset Information")

st.write("Dataset Shape:", df.shape)

st.write("Columns:")

st.write(df.columns)

# =========================================================
# TEMPERATURE TREND
# =========================================================

st.subheader(" Temperature Trend")

fig1, ax1 = plt.subplots(figsize=(12,6))

ax1.plot(
    df['Formatted Date'],
    df['Temperature']
)

ax1.set_xlabel("Date")

ax1.set_ylabel("Temperature")

ax1.set_title("Temperature Trend Over Time")

st.pyplot(fig1)

# =========================================================
# HUMIDITY DISTRIBUTION
# =========================================================

st.subheader(" Humidity Distribution")

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.histplot(
    df['Humidity'],
    bins=30,
    kde=True,
    ax=ax2
)

ax2.set_title("Humidity Distribution")

st.pyplot(fig2)

# =========================================================
# MONTHLY AVERAGE TEMPERATURE
# =========================================================

st.subheader(" Monthly Average Temperature")

monthly_temp = df.groupby('Month')['Temperature'].mean()

fig3, ax3 = plt.subplots(figsize=(10,5))

monthly_temp.plot(
    kind='bar',
    ax=ax3
)

ax3.set_xlabel("Month")

ax3.set_ylabel("Average Temperature")

ax3.set_title("Monthly Average Temperature")

st.pyplot(fig3)

# =========================================================
# CORRELATION HEATMAP
# =========================================================

st.subheader(" Correlation Heatmap")

fig4, ax4 = plt.subplots(figsize=(8,6))

sns.heatmap(
    df[['Temperature', 'Humidity', 'WindSpeed', 'Pressure']].corr(),
    annot=True,
    cmap='coolwarm',
    ax=ax4
)

ax4.set_title("Feature Correlation")

st.pyplot(fig4)

# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

st.subheader(" Machine Learning Model")

# FEATURES
X = df[['Humidity', 'WindSpeed', 'Pressure']]

# TARGET
y = df['Temperature']

# =========================================================
# SPLIT DATASET
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model = LinearRegression()

model.fit(X_train, y_train)

# =========================================================
# PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader(" Model Performance")

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

st.write(" Mean Absolute Error:", mae)

st.write(" R2 Score:", r2)

# =========================================================
# TEMPERATURE PREDICTION
# =========================================================

st.subheader(" Predict Temperature")

humidity = st.slider(
    "Select Humidity",
    0.0,
    1.0,
    0.5
)

windspeed = st.slider(
    "Select Wind Speed",
    0.0,
    50.0,
    10.0
)

pressure = st.slider(
    "Select Pressure",
    900.0,
    1100.0,
    1015.0
)

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("Predict Temperature"):

    sample_data = [[
        humidity,
        windspeed,
        pressure
    ]]

    prediction = model.predict(sample_data)

    st.success(
        f" Predicted Temperature: {prediction[0]:.2f} °C"
    )

# =========================================================
# SAVE MODEL
# =========================================================

with open("weather_model.pkl", "wb") as file:
    pickle.dump(model, file)

# =========================================================
# FOOTER
# =========================================================

st.success(" Dashboard Loaded Successfully")

