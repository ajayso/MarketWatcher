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
from tensorflow.keras.callbacks import EarlyStopping
from logger import logger
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
import sys
import os

#from DataProcessor import DataProcessor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

# plotting & outputs
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from pprint import pprint

#Monitoring on NR 
import newrelic.agent
from monitoring.Manager import MLLogger
newrelic.agent.initialize('newrelic.ini', 'development')
application = newrelic.agent.application()


"""
Appended Models : LSTM , CNN , GAN with LSTM Generator.

This Function returns a Token which recommends the best Model for the given data set.
Input :

dataframe ---> Pandas Dataframe , Cleaned out of NaNs.
Threshold ----> The number of minimum rows in the target datasets ( Integer only )
target ----> target price index (String)
Corr_Thresh ---> Float , for minimum Correlation Score.
timesteps ---> lookback period ( Integer only )

Returns String.

"""
def lr_decay(epoch, lr):
        if epoch != 0 and epoch % 5 == 0:
                return lr * 0.02
        return lr

class PersistModel:
        def __init__(self,name,model):
                self.name = name
                self.model = model
        @newrelic.agent.background_task(name='PersistModel-save', group='Task')
        def Save(self,scriptcode, modelpath,modeltype):
                model_json = self.model.to_json()
                filename = modelpath + "\\" + scriptcode + ".json"
                with open(filename, "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                self.model.save_weights(modelpath + "\\" + scriptcode + ".h5")
        @newrelic.agent.background_task(name='PersistModel-Save', group='Task')
        def Read(self,scriptcode, modelpath):
                filename = modelpath + "\\" + scriptcode + ".json"

                #json_file = open(filename, 'r')
                #json_file.close()
                json_file = open(filename, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights(modelpath + "\\" + scriptcode + ".h5")
                loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                return(loaded_model)

               
                             
class xModel:

        def PCA_n(self,X):#principal component analysis
                scaler = preprocessing.StandardScaler().fit(X)
                X_scaled = scaler.transform(X)
                X_scaled= pd.DataFrame(X_scaled)
                estimator_pca = PCA(n_components=None)
                estimator_pca.fit(X_scaled)
                evr = estimator_pca.explained_variance_ratio_ 
                plt.figure(figsize=(8, 5))
                plt.plot(np.arange(1, len(evr) + 1), np.cumsum(evr*100), "-o")
                plt.title("PCA", fontsize=15)
                plt.xlabel("n_components",fontsize=15)
                plt.ylabel("(%)",fontsize=15)
                plt.xticks(range(1,len(evr),2),fontsize=12)
                plt.yticks(fontsize=12)
                plt.show()
        
        def pcaX(self,X,n):#Data dimensionality reduction
                scaler = preprocessing.StandardScaler().fit(X)
                X_scaled = scaler.transform(X)
                pca = PCA(n_components=n)
                pca_X = pca.fit_transform(X)
                print('降维：',X_scaled.shape,'-->',pca_X.shape)
                return pd.DataFrame(pca_X)

        def __init__(self,script_code,model_name, lookback,features, target,monitor,targetfeatures):
                self.lookback = lookback
                self.features = features
                self.target = target #len(target)
                self.name =model_name
                self.targetfeatures = targetfeatures
                self.script_code = script_code
                earlystopping = EarlyStopping(
                monitor=monitor, 
                patience=0.01, 
                verbose=1, 
                mode='min'
                )
                lr = LearningRateScheduler(lr_decay, verbose=1)
                reduce_lr = ReduceLROnPlateau(
                        monitor=monitor, 
                        factor=0.2,   
                        patience=5, 
                        min_lr=0.001,
                        verbose=2,
                        mode="auto"
                        )
                csv_log = CSVLogger(model_name+ "results.csv")
                

                checkpoint_path = 'model_checkpoints/'
                checkpoint = ModelCheckpoint(
                        filepath=checkpoint_path,
                        save_freq='epoch',
                        save_weights_only=True,
                        verbose=1
                )
                self.callbacks = [earlystopping,lr,reduce_lr,csv_log,checkpoint]
                #self.callbacks = [reduce_lr]

                
                

        @newrelic.agent.background_task(name='xModel-CNNCreate', group='Task')       
        def CNN(self,):
                CNN = Sequential()
                CNN.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(self.lookback, self.features)))
                CNN.add(Conv1D(filters=128,kernel_size=2,activation='relu'))
                CNN.add(MaxPooling1D(2))
                CNN.add(Conv1D(filters=128,kernel_size=1,activation='relu'))
                CNN.add(Conv1D(filters=128,kernel_size=1,activation='relu'))
                CNN.add(layers.TimeDistributed(Dense(1)))
                CNN.add(Flatten())
                CNN.add(Dense(50, activation='relu'))
                CNN.add(Dense(self.target))
                CNN.compile(optimizer='adam', loss='mae',metrics=['accuracy'])
            
        @newrelic.agent.background_task(name='xModel-LSTMCreate', group='Task')  
        def LSTM(self):
                biistm = layers.Bidirectional(layers.LSTM(units = 200,input_shape=(self.lookback,self.features)))
                LSTM = Sequential()
                #LSTM.add((layers.LSTM(units = 200,input_shape=(self.lookback,self.features))))
                LSTM.add(layers.Bidirectional(layers.LSTM(units = 256,input_shape=(self.lookback,self.features))))
                #LSTM.add(layers.Bidirectional(layers.LSTM(10)))
                LSTM.add(Dense(50))
                LSTM.add(Dropout(0.5))
                LSTM.add(Dense(units=self.target , activation = 'relu'))
                LSTM.add(Dropout(0.4))
                LSTM.compile(optimizer='adadelta',loss="mean_absolute_error",metrics=['accuracy'])
                
                
                self.model = LSTM

        @newrelic.agent.background_task(name='xModel-TrainModel', group='Task')  
        def train(self,X_train,Y_train,X_test,Y_test,Model_Array,modelList,sc,sc_predict,
                epochs=500,batch_size=32, verbose=1,
                lastbatch=None,
                sourceColumns=None
                ):

                #ML Monitoring
                ml_logger = MLLogger()
                insert_key = self.name
                metadata = {"environment": "docker", "dataset": self.script_code}
                ml_logger.register_Model(
                        insert_key,
                        self.name,
                        metadata,
                        sourceColumns ,
                        self.targetfeatures,
                        "numeric",
                        "1.0"
                )
                
                #ml_logger.record_interface_data(X_train,Y_train)

                history = self.model.fit(X_train,Y_train,
                validation_data=(X_test,Y_test),
                epochs=500,batch_size=32,verbose=1,
                callbacks=self.callbacks)
                print(self.model.summary())
                #X_val = X_test[1:7,]
                #Y_val = Y_test[7,]
                #ml_logger.drift(X_train,X_test,Y_train,Y_test)
                acc= history.history["accuracy"]
                ml_logger.record_metrics(acc)
                Model_Array[self.name]= self.model.evaluate(X_test,Y_test)
                modelList.append(PersistModel(self.name,self.model))
                #pX = self.model.predict(X_val)
                #print(pX)
                #original_px = sc_predict.inverse_transform(pX)
                #print("Validation of px {}".format(original_px))
                predicted_data = self.model.predict(X_test)
                print("Predictions------")
                print(predicted_data)
                print(predicted_data.shape)
                original_data = sc_predict.inverse_transform(predicted_data)
                print("Predictions------Data")
                print(original_data)
                print("Predictions------Data")
                print(original_data.shape)
                print(self.targetfeatures)
                dfPredicted = pd.DataFrame(original_data, columns = [self.targetfeatures])
                dfPredicted.to_csv("{}-{}-Data.csv".format(self.script_code, self.name))
                print(self.script_code)

                forecast_steps=10 
                # #Prediction of Last Batch
                #dataframe for the original data
                total_features = len(sourceColumns) 
                print(lastbatch)
                origins = lastbatch
                lookback= 7
                forecast = pd.DataFrame(columns=self.targetfeatures)
                # to be repaced by actual no of input features
                for i in range(0,forecast_steps):
                        print(lastbatch.shape)
                        X_batch = origins[i:i+lookback]
                        print(X_batch)
                        X_batch=np.array(X_batch)
                        X_batch_scaled = sc.transform(X_batch)
                        X_batch_scaled=X_batch_scaled.reshape(1,7,total_features)
                        print(X_batch_scaled.shape)
                        batch_predicted_data = self.model.predict(X_batch_scaled)
                        print("Forecast------")
                        
                        print(batch_predicted_data.shape)
                        print(forecast)
                        batch_predicted_data = sc_predict.inverse_transform(batch_predicted_data)
                        print(batch_predicted_data)
                        # add batch_predicted_data to origins
                        forecast.loc[len(forecast.index)] = batch_predicted_data[0]
                        origins.loc[len(origins.index)] = batch_predicted_data[0]
                        print(forecast.shape)
                        #print(origin.shape)
                        
                forecast.to_csv("{}-{}-Y Forecasted-Data.csv".format(self.script_code, self.name))
                #dfLastBatchPredicted = pd.DataFrame(original_data, columns = [self.targetfeatures])
                #origins.to_csv("{}-{}-Forecasted-Data.csv".format(self.script_code, self.name))
                
                
                # data = xModel.Forecast(
                #         lastbatch,
                #         script_code=self.script_code,
                #         name=self.name,
                #         targetfeatures=self.targetfeatures,
                #         sc=sc,
                #         sc_predict=sc_predict
                #  )
                # print(data)
        @newrelic.agent.background_task(name='xModel-ForecastModel', group='Task')  
        def Forecast(X,script_code,name,targetfeatures,sc=None,sc_predict=None):
                print("Inside here---")
                  
                modelpath = os.getcwd() + "\Models" 
                persistedModel = PersistModel(name,None)
                model = persistedModel.Read(scriptcode=script_code,modelpath=modelpath)
                X_forecast=np.array(X)
                #sc = MinMaxScaler(feature_range=(0,1)).fit(X_forecast)
                X_forecast_scaled = sc.transform(X_forecast)
                X_forecast_scaled=X_forecast_scaled.reshape(1,7,19)
                forecast_data = model.predict(X_forecast_scaled)
                print("Forecasted------")
                print(forecast_data)
                print(forecast_data.shape)
                original_data = sc_predict.inverse_transform(forecast_data)
                dfLastBatchForecasted = pd.DataFrame(original_data, columns = [targetfeatures])
                dfLastBatchForecasted.to_csv("{}-{}-Forecasted-Data.csv".format(script_code, name))
                return (original_data)






        

        



                
class ModelManager:

        def TrainMode(self):
                print("Train Model---")
        @newrelic.agent.background_task(name='Forecast', group='Task')
        def Forecast(self,scriptcode,data,lookback,name,modelpath=None):
                xmodel = xModel(script_code=scriptcode,model_name=name,
                lookback=lookback,features="", target=0,monitor="",targetfeatures="")
                #forecast_data = xModel.Forecast(X=data,script_code=scriptcode,name=name,
                #targetfeatures=self.targetfeatures)
                #print(forecast_data)


        
        @newrelic.agent.background_task(name='ModelManager-Selector', group='Task')  
        def Selector(self,scriptcode,data,Threshold,target,Corr_Thresh,split,timesteps,modelpath):
                target=["Close"]
                #data = DataProcessor(dataframe,Threshold,target,Corr_Thresh)
                print("Data shape finally {}".format(data.shape))
                # Last Row specifics
                

                trainin_limit = split
                training_upbound = split*data.shape[0]
                training_upbound = math.ceil(training_upbound)
                self.targetColumns = target
                self.sourceColumns = data.columns
                #Target Indices
                target_col_indices = [data.columns.get_loc(c) for c in target if c in data]
                print("Indices of Target Column={}".format(target_col_indices))

                lookback = timesteps
                features = data.shape[1]
                
                lastiteration = data.tail(lookback)
           
                print(lastiteration)
      
                totalrows = data.shape[0]
                print(data.shape)
                data = data[0:totalrows-lookback]
                print(data.shape)

                
                Model_Array=dict()
                modelList = []

                #Train data and scaling
                training_data = data.iloc[:training_upbound,:]
                test_data = data.iloc[training_upbound+1:,:]
                sc = MinMaxScaler(feature_range=(0,1)).fit(training_data)
                sc_predict = MinMaxScaler(feature_range=(0,1))
                training_data_scaled = sc.fit_transform(training_data)
                training_target_scaled = sc_predict.fit_transform(training_data.iloc[:,target_col_indices]) #.values.reshape(-1,1))
                print("Training target scaled shape {}".format(training_target_scaled.shape))
                
                print("Training data shape {}".format(training_data.shape))
                print("training_data_scaled shape {}".format(training_data_scaled.shape))
                X_train = []
                Y_train = []
                for i in range(lookback,training_data_scaled.shape[0]):
                        X_train.append(training_data_scaled[i-lookback:i,:])
                        Y_train.append(training_data_scaled[i,target_col_indices])      
                X_train,Y_train = np.array(X_train),np.array(Y_train)

                #Test Data and scaling
                dataset_total = pd.DataFrame() #emty dataframe
                dataset_total = training_data.iloc[-lookback:,:]
                dataset_total = pd.concat([dataset_total ,data.iloc[training_upbound+1:,:]],axis=0)
                inp = dataset_total.copy()
                print("Inp shape {}".format(inp.shape))
                inp = sc.transform(inp)
                X_test = []
                Y_test = []
                for i in range(lookback,dataset_total.shape[0]):
                        X_test.append(inp[i-lookback:i,:])
                        Y_test.append(inp[i,target_col_indices])
                X_test,Y_test = np.array(X_test),np.array(Y_test)

                print("X_train shape=={}".format(X_train.shape))
                print("Y_train shape=={}".format(Y_train.shape))
                print("X_test shape=={}".format(X_test.shape))
                print("Y_test shape=={}".format(Y_test.shape))

        
                print("LSTM is being trained and tested now\n")
                target = Y_train.shape[1]
                print("Target is {}".format(target))
                
                xmodel = xModel(script_code = scriptcode,lookback=lookback,features=features,target=target,monitor="val_loss",model_name="LSTM",targetfeatures=self.targetColumns)
                
                #xmodel.PCA_n(data)
                xmodel.LSTM()
                xmodel.train(X_train,Y_train,X_test,Y_test,Model_Array,modelList,sc,sc_predict,
                epochs=500,batch_size=16, verbose=1,
                lastbatch=lastiteration,
                sourceColumns = self.sourceColumns
                )
                        
                

                print("CNN is being trained and tested now\n")
                #CNN
                xmodel.name="CNN"
                xmodel.CNN()
                xmodel.train(X_train,Y_train,X_test,Y_test,Model_Array,modelList,sc,sc_predict,
                epochs=500,batch_size=16, verbose=1,
                lastbatch=lastiteration,
                sourceColumns = self.sourceColumns
                )
                
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
