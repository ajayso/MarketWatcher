import pandas as pd
import numpy as np

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import CuDNNLSTM,CuDNNGRU
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import model_from_json
from logger import logger

#from DataProcessor import DataProcessor
from sklearn.preprocessing import MinMaxScaler

"""
Appended Models : LSTM , CNN , GAN with LSTM Generator.

This Function returns a Token which recommends the best Model for the given data set.
Input :

dataframe ---> Pandas Dataframe , Cleaned out of NaNs.
Threshold ----> The number of minimum rows in the target datasets ( Integer only )
target ----> target price index (String)
Corr_Thresh ---> Float , for minimum Correlation Score.
timesteps ---> lookback period ( Interger only )

Returns String.

"""
class PersistModel:
        def __init__(self,name,model):
                self.name = name
                self.model = model

        def Save(self,scriptcode, modelpath,modeltype):
                model_json = self.model.to_json()
                filename = modelpath + "\\" + scriptcode + ".json"
                with open(filename, "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                self.model.save_weights(modelpath + "\\" + scriptcode + ".h5")
        def Read(self,scriptcode, modelpath):
                filename = modelpath + "\\" + scriptcode + ".json"
                json_file = open(filename, 'r')
                json_file.close()
                loaded_model.load_weights(modelpath + "\\" + scriptcode + ".h5")
                loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

               
                               
                
                
class ModelManager:

        def Selector(self,scriptcode,data,Threshold,target,Corr_Thresh,split,timesteps,modelpath):

                #data = DataProcessor(dataframe,Threshold,target,Corr_Thresh)
                trainin_limit = split
                training_upbound = split*data.shape[0]
                training_upbound = math.ceil(training_upbound)

                target = list(data.columns).index(target)

                lookback = timesteps
                features = data.shape[1]
                
                Model_Array=dict()
                modelList = []

                #Train data and scaling
                training_data = data.iloc[:training_upbound,:]
                test_data = data.iloc[training_upbound+1:,:]
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

                #Test Data and scaling
                dataset_total = pd.DataFrame() #emty dataframe
                dataset_total = training_data.iloc[-lookback:,:]
                dataset_total = pd.concat([dataset_total ,data.iloc[training_upbound+1:,:]],axis=0)
                inp = dataset_total.copy()
                inp = sc.transform(inp)
                X_test = []
                Y_test = []
                for i in range(lookback,dataset_total.shape[0]):
                        X_test.append(inp[i-lookback:i,:])
                        Y_test.append(inp[i,target])
                X_test,Y_test = np.array(X_test),np.array(Y_test)

                print("LSTM is being trained and tested now\n")
                #LSTM training structure
                LSTM = Sequential()
                LSTM.add(layers.LSTM(units = 200,input_shape=(lookback,features)))
                LSTM.add(Dense(units=1 , activation = 'linear'))
                LSTM.compile(optimizer='adadelta',loss="mean_absolute_error")
                LSTM.fit(X_train,Y_train,epochs=500,batch_size=16,verbose=1)
                Model_Array['LSTM']= LSTM.evaluate(X_test,Y_test)
                modelList.append(PersistModel('LSTM',LSTM))
                predicted_LSTM = LSTM.predict(X_test)
                predicted_LSTM = sc_predict.inverse_transform(predicted_LSTM)

                print("CNN is being trained and tested now\n")
                #CNN
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
                Model_Array['CNN'] = CNN.evaluate(X_test,Y_test)
                modelList.append(PersistModel('CNN',CNN))
                predicted_CNN = CNN.predict(X_test)
                predicted_CNN = sc_predict.inverse_transform(predicted_CNN)
                
                def generator():

                        gen = Sequential()
                        gen.add(layers.LSTM(200,input_shape=(lookback,features)))
                        gen.add(Dense(1,activation='linear'))
                        return gen

                def new_generator():
                
                        gcnn = Sequential()
                        gcnn.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(lookback, features)))
                        gcnn.add(Conv1D(filters=128,kernel_size=2,activation='relu'))
                        gcnn.add(MaxPooling1D(2))
                        gcnn.add(Conv1D(filters=128,kernel_size=1,activation='relu'))
                        gcnn.add(Conv1D(filters=128,kernel_size=1,activation='relu'))
                        gcnn.add(Flatten())
                        gcnn.add(Dense(50, activation='relu'))
                        gcnn.add(Dense(1))
                        
                        return gcnn

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

                Generator_CNN = new_generator()
                Generator_CNN.compile(loss='mae',optimizer="adam")

                Discriminator = discriminator()
                Discriminator.compile(loss='mse',optimizer="adam")

                stacked = stacked_generator_discriminator(Discriminator,Generator)
                stacked.compile(loss='mae',optimizer='adam')

                stacked_CNN = stacked_generator_discriminator(Discriminator,Generator_CNN)
                stacked_CNN.compile(loss='mae',optimizer='adam')
                
                epochs = 6000
                batch =16
                PYTHONHASHSEED=0
                np.random.seed=1

                print("GAN - LSTM is being Trained and Tested now\n")

                for count in range(epochs):
                        random_index = np.random.randint(0,len(X_train)-batch/2)
                        gen_data = Generator.predict(X_train[random_index:random_index+batch//2])
                        gen_data = gen_data.reshape((batch//2,))
                        x_combined_batch = np.concatenate((Y_train[random_index:random_index+batch//2], gen_data))
                        y_combined_batch = np.concatenate((Y_train[random_index:random_index+batch//2],gen_data))
                        d_loss= Discriminator.train_on_batch(x_combined_batch,y_combined_batch)

                        g_loss = stacked.train_on_batch(X_train[random_index:random_index+batch],Y_train[random_index:random_index+batch])
                        logger.info('epoch: {}, [Discriminator: {}], [Generator: {}]'.format(count,d_loss,g_loss))

                Model_Array['GAN-LSTM']= Generator.evaluate(X_test,Y_test)
                modelList.append(PersistModel('GAN-LSTM',Generator))
                predicted_GAN = Generator.predict(X_test)
                predicted_GAN = sc_predict.inverse_transform(predicted_GAN)

                print("GAN - LSTM is being Trained and Tested now\n")

                epochs = 6000
                batch =16
                PYTHONHASHSEED=0
                np.random.seed=1

                for count in range(epochs):
                        random_index = np.random.randint(0,len(X_train)-batch/2)
                        gen_data = Generator_CNN.predict(X_train[random_index:random_index+batch//2])
                        gen_data = gen_data.reshape((batch//2,))
                        x_combined_batch = np.concatenate((Y_train[random_index:random_index+batch//2], gen_data))
                        y_combined_batch = np.concatenate((Y_train[random_index:random_index+batch//2],gen_data))
                        d_loss= Discriminator.train_on_batch(x_combined_batch,y_combined_batch)
                        
                        g_loss = stacked_CNN.train_on_batch(X_train[random_index:random_index+batch],Y_train[random_index:random_index+batch])
                        logger.info('epoch: {}, [Discriminator: {}], [Generator: {}]'.format(count,d_loss,g_loss))

                Model_Array['GAN-CNN']= Generator_CNN.evaluate(X_test,Y_test)
                modelList.append(PersistModel('GAN-CNN',Generator))
                predicted_GAN_CNN = Generator_CNN.predict(X_test)
                predicted_GAN_CNN = sc_predict.inverse_transform(predicted_GAN_CNN)

                
                
                print(Model_Array)
                best_model = min(Model_Array, key=Model_Array.get)
                for persistModel in modelList:
                    if (best_model is persistModel.name):
                            print("best model is " + best_model)
                            persistModel.Save(scriptcode,modelpath,best_model)
                
                print("\n")
                print("#############################################")
                print("Best Model with the Current Data -->",best_model)      
                return best_model
