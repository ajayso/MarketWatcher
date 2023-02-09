import pandas as pd
import numpy as np

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LeakyReLU
from logger import logger
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Monitoring 
from monitoring.Manager import MLLogger


#from DataProcessor import DataProcessor
class ModelBuilder:
	def Trainer(self,token,data,Threshold,target,Corr_Thresh,timesteps):

		
		#data = DataProcessor(dataframe,Threshold,target,Corr_Thresh)
		trainin_limit = 1

		target = list(data.columns).index(target)

		lookback = timesteps
		features = data.shape[1]

		training_data = data.iloc[:,:]
		sc = MinMaxScaler(feature_range=(0,1))
		sc_predict = MinMaxScaler(feature_range=(0,1))
		training_data_scaled = sc.fit_transform(training_data)
		training_target_scaled = sc_predict.fit_transform(training_data.iloc[:,target].values.reshape(-1,1))

		X_train = []
		Y_train = []
		for i in range(lookback,training_data.shape[0]):
			X_train.append(training_data_scaled[i-lookback:i,:])
			Y_train.append(training_data_scaled[i,target])      
		X_train,Y_train = np.array(X_train),np.array(Y_train)

		if(token == 'LSTM'):
			print("Token is",token,"and now commencing training on the dataset... \n")
			#LSTM training structure
			LSTM = Sequential()
			LSTM.add(layers.LSTM(units = 200,input_shape=(lookback,features)))
			LSTM.add(Dense(units=1 , activation = 'linear'))
			LSTM.compile(optimizer='adadelta',loss="mean_absolute_error")
			LSTM.fit(X_train,Y_train,epochs=500,batch_size=16,verbose=1)
			self.model = LSTM

		elif(token=='CNN'):
			print("Token is",token,"and now commencing training...")
			CNN = Sequential()
			CNN.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(lookback, features)))
			CNN.add(Conv1D(filters=128,kernel_size=2,activation='relu'))
			CNN.add(MaxPooling1D(2))
			CNN.add(Conv1D(filters=128,kernel_size=1,activation='relu'))
			CNN.add(Conv1D(filters=128,kernel_size=1,activation='relu'))
			CNN.add(Flatten())
			CNN.add(Dense(50, activation='relu'))
			CNN.add(Dense(1))
			CNN.compile(optimizer='adam', loss='mae')
			CNN.fit(X_train,Y_train,epochs=500,verbose=1,batch_size=16)
			self.model = CNN

		elif(token =='GAN-LSTM'):
			print("Token is",token,"and now commencing training...")
			def generator():

				gen = Sequential()
				gen.add(layers.LSTM(200,input_shape=(lookback,features)))
				gen.add(Dense(1,activation='linear'))
				return gen

			def discriminator():

				model = Sequential()
				model.add(Dense((10), input_shape=(1,)))
				model.add(LeakyReLU(alpha=0.2))
				model.add(Dense(int((10) / 2)))
				model.add(LeakyReLU(alpha=0.2))
				model.add(Dense(1, activation='linear'))

				return model

			def stacked_generator_discriminator(D,G):
				D.trainable = False
				model = Sequential()
				model.add(G)
				model.add(D)
				return model

			Generator = generator()
			Generator.compile(loss='mae',optimizer="adam")

			Discriminator = discriminator()
			Discriminator.compile(loss='mse',optimizer="adam")

			stacked = stacked_generator_discriminator(Discriminator,Generator)
			stacked.compile(loss='mae',optimizer='adam')
			
			epochs = 6000
			batch =16
			PYTHONHASHSEED=0
			np.random.seed=1

			for count in range(epochs):
				random_index = np.random.randint(0,len(X_train)-batch/2)
				gen_data = Generator.predict(X_train[random_index:random_index+batch//2])
				gen_data = gen_data.reshape((batch//2,))
				x_combined_batch = np.concatenate((Y_train[random_index:random_index+batch//2], gen_data))
				y_combined_batch = np.concatenate((Y_train[random_index:random_index+batch//2],gen_data))
				d_loss= Discriminator.train_on_batch(x_combined_batch,y_combined_batch)

				g_loss = stacked.train_on_batch(X_train[random_index:random_index+batch],Y_train[random_index:random_index+batch])
				logger.info('epoch: {}, [Discriminator: {}], [Generator: {}]'.format(count,d_loss,g_loss))
			self.model = Generator

		elif(token =='GAN-CNN'):
			print("Token is",token,"and now commencing training...")
			def generator():
				gen = Sequential()
				gen.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(lookback, features)))
				gen.add(Conv1D(filters=128,kernel_size=2,activation='relu'))
				gen.add(MaxPooling1D(2))
				gen.add(Conv1D(filters=128,kernel_size=1,activation='relu'))
				gen.add(Conv1D(filters=128,kernel_size=1,activation='relu'))
				gen.add(Flatten())
				gen.add(Dense(50, activation='relu'))
				gen.add(Dense(1))
				
				return gen

			def discriminator():

				model = Sequential()
				model.add(Dense((10), input_shape=(1,)))
				model.add(LeakyReLU(alpha=0.2))
				model.add(Dense(int((10) / 2)))
				model.add(LeakyReLU(alpha=0.2))
				model.add(Dense(1, activation='linear'))

				return model

			def stacked_generator_discriminator(D,G):
				D.trainable = False
				model = Sequential()
				model.add(G)
				model.add(D)
				return model

			Generator = generator()
			Generator.compile(loss='mae',optimizer="adam")

			Discriminator = discriminator()
			Discriminator.compile(loss='mse',optimizer="adam")

			stacked = stacked_generator_discriminator(Discriminator,Generator)
			stacked.compile(loss='mae',optimizer='adam')
			
			epochs = 6000
			batch =16
			PYTHONHASHSEED=0
			np.random.seed=1

			for count in range(epochs):
				random_index = np.random.randint(0,len(X_train)-batch/2)
				gen_data = Generator.predict(X_train[random_index:random_index+batch//2])
				gen_data = gen_data.reshape((batch//2,))
				x_combined_batch = np.concatenate((Y_train[random_index:random_index+batch//2], gen_data))
				y_combined_batch = np.concatenate((Y_train[random_index:random_index+batch//2],gen_data))
				d_loss= Discriminator.train_on_batch(x_combined_batch,y_combined_batch)

				g_loss = stacked.train_on_batch(X_train[random_index:random_index+batch],Y_train[random_index:random_index+batch])
				logger.info('epoch: {}, [Discriminator: {}], [Generator: {}]'.format(count,d_loss,g_loss))
			self.model = Discriminator
		return(self.model)