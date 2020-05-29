import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_yearly
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import holidays
import csv
#import pandas_profiling as pp
#import matplotlib.pyplot as plt
#from fbprophet import Prophet

#matplotlib inline
 
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')

#time = df_fullset['timestamp']
#s = time.replace('D',' ', regex = True)
df_fullset = pd.read_csv("interim_data1.csv")
df_model=pd.DataFrame()

df_model['y']=df_fullset['close'] #rename the columns
df_model['ds']=df_fullset['timestamp']

#print(type(df_model['ds']))
        
df_model['ds'] = pd.to_datetime(df_model.ds, errors='ignore', format='%Y-%m-%d %H:%M:%S') #converting series to datetime object
df_model['ds'] = df_model['ds'].dt.tz_convert(None) #remove timezone
#df_model['y'] = np.log(df_model['y'])
#df_model.set_index('ds').y.plot()
#plt.show()

m = Prophet(seasonality_mode = "multiplicative", growth = "linear", interval_width = 0.75, daily_seasonality = False, weekly_seasonality= False, yearly_seasonality = False, changepoint_range=0.95, changepoint_prior_scale=0.1, seasonality_prior_scale = 25) #initialize prophet
m.add_seasonality(name='daily', period=1, fourier_order=15) #prior_scale=0.1)
#m.add_seasonality(name='weekly', period=7, fourier_order=3) #prior_scale=0.1)
m.add_seasonality(name='yearly', period=365.25, fourier_order=10) #prior_scale=0.1)
m.add_country_holidays(country_name='US')
#m.add_seasonality(name=’monthly’, period=30.5, fourier_order=5, prior_scale=0.02)
m.fit(df_model) #fit it
future = m.make_future_dataframe(periods=1, include_history = True, freq = 'd') #command for setting up predictions to be made
#print(future.tail())

forecast = m.predict(future) #predicting future
r = print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1), file=open("output1.txt", "a"))
df = pd.read_fwf('output1.txt')
df.to_csv('results1.csv')

#with open("results.csv", "wt") as fp:
 #   writer = csv.writer(fp, delimiter=",")
    #for row in data:
  #  writer.writerow(r)
#print(forecast.tail())
"""
def mean_absolute_percentage_error(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    return np.mean(np.abs((y-yhat)/y))* 100
results = cross_validation(m)
mape_baseline = mean_absolute_percentage_error(results.y, results.yhat)
print(mape_baseline)
"""
#f = cross_validation(m, horizon = '30 days', initial=' 500 days', period='30 days') 
#print(f[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(30))

#performance metrics
#p = performance_metrics(f)
#print(p.tail(30))

#fig1 = m.plot(forecast) #plotting
#fig2 = m.plot_components(forecast)
#plt.title("Facbook Prophet Forecast and Fitting")
#a = add_changepoints_to_plot(fig1.gca(), m, forecast)
#plt.show()

"""
#visualizing the results

df_model.set_index('ds', inplace=True)
forecast.set_index('ds', inplace=True) 
viz_df = df_fullset.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
#print(viz_df.tail())

#viz_df['', ''].plot()
#plt.show()

fig, ax1 = plt.subplots()
ax1.plot(viz_df.close)
ax1.plot(viz_df.yhat, color='black', linestyle=':')
ax1.fill_between(viz_df.index, np.exp(viz_df['yhat_upper']), np.exp(viz_df['yhat_lower']), alpha=0.5, color='darkgray')
ax1.set_title('Close price (Orange) vs Prophet Forecast (Black)')
ax1.set_ylabel('Bitcoin Price')
ax1.set_xlabel('Date')

L=ax1.legend() #get the legend
L.get_texts()[0].set_text('Actual price') #change the legend text for 1st plot
L.get_texts()[1].set_text('Forecasted price') #change the legend text for 2nd plot
plt.show()
"""
