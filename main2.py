import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Dummy data generation for Emergency, Ward, and Clinic admissions
def generate_dummy_data(start_date):
    today = date.today()
    num_days = (today - start_date).days + 1  # Include today
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Simulating admissions for each category using Poisson distribution
    emergency_admissions = np.random.poisson(lam=10, size=num_days)  # Average of 10 emergency admissions per day
    ward_admissions = np.random.poisson(lam=15, size=num_days)  # Average of 15 ward admissions per day
    clinic_admissions = np.random.poisson(lam=20, size=num_days)  # Average of 20 clinic admissions per day

    # Create a DataFrame for all the categories
    data = pd.DataFrame({
        'Date': dates,
        'Emergency': emergency_admissions,
        'Ward': ward_admissions,
        'Clinic': clinic_admissions
    })
    return data

# Streamlit app setup
st.title('Patient Admission Prediction')

# Sidebar input for the type of admission (Emergency, Ward, or Clinic)
st.sidebar.header("Select Admission Type")
admission_type = st.sidebar.selectbox('Choose the admission type:', ['Emergency', 'Ward', 'Clinic'])

# Sidebar input for the number of years to forecast
st.sidebar.header("Forecast Parameters")
n_years = st.sidebar.slider('Years of prediction:', 1, 4, 1)  # Default value set to 1 year

# Define start date for dummy data
start_date = date(2022, 1, 1)  # Starting from January 1, 2022

# Generate dummy data
data = generate_dummy_data(start_date)

# Display the latest data for the selected admission type in the sidebar
st.sidebar.subheader(f'Latest {admission_type} Data')
st.sidebar.write(data[['Date', admission_type]].tail())

# Main content: Display admissions data for the selected type
st.subheader(f'{admission_type} Admissions Data')
st.write(data[['Date', admission_type]].tail())

# Plot raw data for the selected category
def plot_raw_data(data, admission_type):
    fig = go.Figure()

    # Plot the selected admission type
    fig.add_trace(go.Scatter(x=data['Date'], y=data[admission_type], name=f"{admission_type} Admissions"))
    
    fig.layout.update(title_text=f'Daily {admission_type} Admissions Over Time', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(data, admission_type)

# Prepare data for Prophet model based on the selected admission type
df_selected = data[['Date', admission_type]].rename(columns={"Date": "ds", admission_type: "y"})

# Create and fit the Prophet model for the selected admission type
m = Prophet()
m.fit(df_selected)

# Create future dataframe and make predictions for the next n_years
num_future_days = n_years * 365
future = m.make_future_dataframe(periods=num_future_days)
forecast = m.predict(future)

# Show and plot forecast for the selected admission type
st.subheader(f'{admission_type} Forecast Data')
st.write(forecast.tail())

st.write(f'{admission_type} Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write(f"{admission_type} Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
