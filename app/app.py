from flask import Flask,render_template,url_for,request
from twilio.rest import Client
app = Flask(__name__, template_folder='templates')
app.static_folder = 'static'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as pdr
#import yahoo finance api fixer
import fix_yahoo_finance as fyf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		#request stock ticker that user inputs
		ticker = request.form['ticker']
		data = [ticker.upper()]
		fyf.pdr_override()
		#read in training set
		dataset_train = pdr.get_data_yahoo(data, start='2015-01-01', end=datetime.now())
		training_set=[]
		#Timestep=60 days
		for i in range(0, len(dataset_train)-61):
			training_set.append(dataset_train["Open"][i])
		training_set=np.array(training_set)
		#normalize data
		from sklearn.preprocessing import MinMaxScaler
		training_set=training_set.reshape(-1,1)
		sc = MinMaxScaler(feature_range = (0, 1))
		training_set_scaled = sc.fit_transform(training_set)
		#separate training set into input and output categories
		X_train = []
		y_train = []
		for i in range(60, 1087):
			X_train.append(training_set_scaled[i-60:i, 0])
			y_train.append(training_set_scaled[i, 0])
		X_train, y_train = np.array(X_train), np.array(y_train)
		#reshape training data
		X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
		#import necessary neural network packages
		from keras.models import Sequential
		from keras.layers import Dense
		from keras.layers import LSTM
		from keras.layers import Dropout
		#create model class
		regressor = Sequential()

		# Adding the first LSTM layer and some Dropout regularization
		regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
		regressor.add(Dropout(0.2))

		# Adding a second LSTM layer and some Dropout regularization
		regressor.add(LSTM(units = 50, return_sequences = True))
		regressor.add(Dropout(0.2))

		# Adding a third LSTM layer and some Dropout regularization
		regressor.add(LSTM(units = 50, return_sequences = True))
		regressor.add(Dropout(0.2))

		# Adding a fourth LSTM layer and some Dropout regularization
		regressor.add(LSTM(units = 50))
		regressor.add(Dropout(0.2))

		# Adding the output layer
		regressor.add(Dense(units = 1))
		#add optimizer and cost function to model
		regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
		#train the model
		regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
		#test on user input data
		new_df = pdr.get_data_yahoo(data, start="2020-03-01", end=datetime.now())
		x_test2=[]
		#Again, timestep=60 days, so I just picked an arbitrary value for start date ("2019-03-01"); all I care about is getting the last 60 days of data and using that info to predict stock price tomorrow
		for i in range(len(new_df)-60, len(new_df)):
			x_test2.append(new_df["Open"][i])
		x_test2=np.array(x_test2)
		x_test2=x_test2.reshape(-1,1)
		x_test2=sc.transform(x_test2)
		X_Test2=[]
		X_Test2.append(x_test2[0:60, 0])
		X_Test2 = np.array(X_Test2)
		X_Test2 = np.reshape(X_Test2, (X_Test2.shape[0], X_Test2.shape[1], 1))
		predicted_stock_price = regressor.predict(X_Test2)
		predicted_stock_price = sc.inverse_transform(predicted_stock_price)
		#send text message to user that includes predicted opening stock price for tomorrow for the specified stock
		number = request.form['number']
		data2 = [number.upper()]
		message="Predicted "+str(data)+" stock opening price for tomorrow: $"+str(predicted_stock_price)
		account_sid = "ACCOUNT_SID"
		auth_token = "AUTH_TOKEN"
		client = Client(account_sid, auth_token)
		client.messages.create(to=data2,from_="TWILIO_NUMBER",body=message)
		#output message to user on the home webpage letting them know that the text message with the prediction has been sent
	return render_template('index.html',prediction_text="Text message with the predicted "+str(data)+" stock price for tomorrow has been sent!")
	
