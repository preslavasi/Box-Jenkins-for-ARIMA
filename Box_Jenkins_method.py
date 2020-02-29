# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:20:23 2019

@author: Preslava Ivanova, 11223
"""
import pandas as pd
from matplotlib import pyplot
from numpy import log
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import numpy
from sklearn.metrics import mean_squared_error
from math import sqrt

# Importing time series data
data=pd.read_csv('Electric_Production.csv')
X=data.values[:,1]
X=X.astype('float64')
print('Plot Data')
pyplot.plot(X)
pyplot.show()

# PART 1 - Identification of the ARIMA model
# Differentiating the data
diffX = log(X)
pyplot.plot(X)
pyplot.show()
X=log(diffX)
pyplot.plot(X)
pyplot.show()
# ACF
print('ACF Original Data')
autocorrelation_plot(X)
pyplot.show()
lag_acf = acf(X)
pyplot.figure()
plot_acf(lag_acf)
pyplot.title('Autocorrelation Function')
pyplot.show()
# PACF
lag_pacf = pacf(X)
plot_pacf(lag_pacf, method='ywm')
pyplot.title('Partial Autocorrelation Function')
pyplot.show()

# PART 2 - Applying the ARIMA model and estimating different criteria
order=(2,2,0)
print ('START ARIMA',order,' for Electric production ')
model = ARIMA(X, order)
model_fit = model.fit(disp=0)
print(model_fit.summary())
# Plotting residual errors
residuals = DataFrame(model_fit.resid)
print('Plot Residuals')
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())
# Durbin Watson
DW=sm.stats.durbin_watson(residuals)
print('DW=', DW) # ('DW=', array([2.46524428])) - near 2, so no autocorrelation

# PART 3 - Plotting residual errors - ACF and PACF
print ('Autocorrelation plot for Residuals and ARIMA',order,' for Electric production data ')
lag_acf = acf(residuals)
plot_acf(lag_acf)
pyplot.show()
print ('Partial Autocorrelation plot for Residuals and ARIMA',order,' for Electric production data ')
lag_pacf = pacf(residuals)
plot_pacf(lag_pacf, method='ywm')
pyplot.show()
# Calculating autocorrelation, Q-statistic and Probability>Q-statistic
r,q,p = sm.tsa.acf(residuals, qstat=True)
datanew = numpy.c_[range(1,41), r[1:], q, p]
table = DataFrame(datanew, columns=['lag', "AC", "Q", "Prob(>Q)"])
print table.set_index('lag') # values 1-12 are under 0.05, others are 0 

# PART 4 - Computing predictions
size = int(15)
train = X[0:size] # train set
test = X[size:len(X)] # test set
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('=%i, predicted=%f, expected(real)=%f' % (size+t,yhat, obs))
# Calculating Mean Forecast Error
MFE = (predictions-test).mean()
print ("MFE = ",MFE) # ('MFE = ', 2.631673892829206e-05)
# Calculating Mean Absolute Error
MAE = (numpy.abs((predictions-test).mean()) / predictions).mean()
print ("MAE = ", MAE) # ('MAE = ', -2.92373649193483e-05)
# Calculating RMSE
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse) # Test RMSE: 0.031
