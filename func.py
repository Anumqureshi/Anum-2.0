import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_yearly
import holidays

def fbp(df):
    df_fullset = pd.read_csv(df)
    df_model=pd.DataFrame()
    df_model['y']=df_fullset['close']
    
    df_model['ds']=df_fullset['timestamp']
    df_model['ds'] = pd.to_datetime(df_model.ds, errors='ignore', format='%Y-%m-%d %H:%M:%S')
    df_model['ds'] = df_model['ds'].dt.tz_convert(None)
    m = Prophet(seasonality_mode = "multiplicative", growth = "linear", interval_width = 0.75, daily_seasonality = False, weekly_seasonality= False, yearly_seasonality = False, changepoint_range=0.95, changepoint_prior_scale=0.1, seasonality_prior_scale = 25)
    m.add_seasonality(name='daily', period=1, fourier_order=15)
    m.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    m.add_country_holidays(country_name='US')
    m.fit(df_model)
    future = m.make_future_dataframe(periods=60, include_history = True, freq = 'd')
    forecast = m.predict(future)
    r = print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(60))
    return 
