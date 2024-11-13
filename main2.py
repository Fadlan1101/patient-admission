# Required libraries
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Dummy data generation until today
def generate_dummy_data(start_date):
    today = date.today()
    num_days = (today - start_date).days + 1  # Include today
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    admissions = np.random.poisson(lam=20, size=num_days)  # Average of 20 admissions per day
    data = pd.DataFrame({'Date': dates, 'Admissions': admissions})
    return data

# Streamlit app setup
st.title('Patient Admission Prediction')

# Define start date for dummy data
start_date = date(2022, 1, 1)  # Starting from January 1, 2022

# Generate dummy data
data = generate_dummy_data(start_date)

st.subheader('Admissions Data')
st.write(data.tail())

# Plot raw data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Admissions'], name="Daily Admissions"))
    fig.layout.update(title_text='Daily Patient Admissions Over Time', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(data)

# Prepare data for Prophet
df_train = data.rename(columns={"Date": "ds", "Admissions": "y"})

# Create and fit the Prophet model
m = Prophet()
m.fit(df_train)

# Create future dataframe and make predictions for the next n_years
n_years = st.slider('Years of prediction:', 1, 4)
num_future_days = n_years * 365
future = m.make_future_dataframe(periods=num_future_days)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast Data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
