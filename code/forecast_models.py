# -*- coding: utf-8 -*-
"""
Module for power demand forecast of the DeForMaL software

contains the classes: 
    NN_Model: Neural Network model for power demand forecasting

"""

__author__     = "Jethro Betcke"
__copyright__  = "Copyright 2024, Jethro Betcke"
__version__    = "0.02"
__maintainer__ = "Jethro Betcke"


from tensorflow import keras
from keras import models 
from keras import layers 
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.initializers import glorot_uniform, glorot_normal
from keras.initializers import he_uniform, he_normal
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import prepdata




class NN_Model():
    def __init__(self,x_columns, hidden_nodes='mean_in_out', hidden_layers=1,
                  activation4hidden='linear'):
        
        """
        prepares the object and attributes:
        
        Args:
        -------
        x_columns, list or pandas column index
            The names of the x variables in the model, is used to determine
            the architecture of the network and to check training inputs 
            for consistency
       
            
        Keyword Args:
        -------------
        hidden_nodes, integer or string, default: 'mean_in_out' 
            determines the number of nodes in the hidden layers
            see method calculate_hidden_nodes
        hidden_layers, integer, default: 1
            number of hidden layers
        activation_func4hidden, string, default: 'linear' 
            activation function for the hidden layers
            see dir(keras.activations) for a list of available activation 
            functions
            
            
            
        Creates Attributes:
        x_data_train_df: pandas dataframe
            temporal intersection of x_data_trai nwith y_data_train
        x_data_train_df: pandas dataframe
            temporal intersection of x_data_trai nwith y_data_train        
        """

        

        #TODO test if common period is long enough
        self.input_params = list(x_columns)
        
        
        # set the NN model parameters
        self.nr_in_nodes = len( self.input_params )
        self.nr_out_nodes = 1 
        self.calculate_hidden_nodes(hidden_nodes)
        self.nr_hidden_layers = hidden_layers
        self.activation4hidden = activation4hidden
        
        self.define_model()
        self.histories = []
        self.x_scaler = None
        self.y_scaler = None
       
              
    
    def calculate_hidden_nodes(self,node_calc_method):
        """
        
        determines the number of nodes in the hidden layer
        in the following manners depending on the value of self.hidden_nodes,
        according to some come rules of thumb:
            
            
        if self.hidden_nodes:
            - is an integer use this value
            - equals 'mean_in_out', take the average of the
              nr of in nodes and the number of out nodes, and round upwards
            - equals 'sqrt_in_out' take the square root of the
              product of the nr of nodes in the input and output layers
            - equals 'equal2in set the nr of hidden nodes equal to the number
              of notes of the input layer
            - equals 'two3thin' 
        
        

        Raises
        ------
        Exception
            when the input on the calculation method is not defined in the 
            function

        Returns
        -------
        None.

        """
        
        #node_rules=['mean_in_out', 'sqrt_in_out', 'equal2in', 'two3thin']
    
        if type(node_calc_method) == int:
            self.nr_hidden_nodes = node_calc_method
        elif node_calc_method == 'mean_in_out':
            self.nr_hidden_nodes = \
                    int(math.ceil( (self.nr_out_nodes + self.nr_in_nodes)/2) )
        elif node_calc_method == 'sqrt_in_out':    
            self.nr_hidden_nodes = \
                int(math.ceil( math.sqrt(self.nr_out_nodes * self.nr_in_nodes) ))
        elif node_calc_method == 'equal2in' :
            self.nr_hidden_nodes = self.nr_in_nodes
        elif node_calc_method == 'two3thin':
            self.nr_hidden_nodes = int(math.ceil(2*self.nr_in_nodes/3))            
        else:           
            raise Exception('unknown way to calculate nodes:'
                                                       f' {self.hidden_nodes}')            
            
    def define_model(self):
        """
        Defines the neural network model, based on the keras Sequential model.
        
        
        Uses Attributes:
        ----------------    
        .x_data_train_df pandas dataframe
            used to determine the number of inputs
        nr_hidden_layers, int
            the number of hidden layers of the model
        nr_hidden_nodes, int
            number of nodes for each hidden layer
        activation4hidden, string or function
            activation function for the hidden layers

        Creates/overwrites Attribute
        -------
        model, keras model

        """
        nr_of_x_params = len (self.input_params)
        
        self.model = keras.Sequential()
        self.model.add( keras.Input(shape=(nr_of_x_params,)) )
        for i in range(self.nr_hidden_layers):  
            print('i: ', i)   
            self.model.add( keras.layers.Dense(units=self.nr_hidden_nodes, 
                                          activation=self.activation4hidden,
                                          kernel_initializer='he_uniform') )
        self.model.add( keras.layers.Dense(1, activation ="linear") )
                
         
        
        
    def train_model(self, x_train_df, y_train_df, x_val_df, y_val_df,
                    learning_rate=0.01, epochs=100, batch_size=1000,
                     verbose=1):
        
        """
        Train the model 
        
        x_train_df: pandas dataframe or pandas series
            the independent, i.e. input, variables for training the model
        y_train_df: pandas dataframe or pandas series
            the dependent, i.e. output, variable for training the model  
            
        x_val_df: pandas dataframe or pandas series
                the independent, i.e. input, variables for validating the model
        y_val_df: pandas dataframe or pandas series
                the dependent, i.e. output, variable for validating the model      
        
        
        Keyword Args:
        ------------    
        
        
        verbose, integer in [0,1,2] default 1
        leval of verbosity of training, 0 is silent, 1 shows progress bar, 
        2 is detailed
        """
        

        
        
        x_train_intersect, y_train_intersect=prepdata.intersect_df(x_train_df, 
                                                                   y_train_df)
        
        
        # first training, determine the scaling for x and y
        if len(self.histories) == 0:
            self.x_scaler = MinMaxScaler()
            x_train_scaled = self.x_scaler.fit_transform(
                                                      x_train_intersect.values)
            self.y_scaler = MinMaxScaler()
            y_train_scaled = self.y_scaler.fit_transform(
                                                      y_train_intersect.values)   
                  
        #subsequent training, only apply the scaling.
        else:
            x_train_scaled = self.x_scaler.transform(x_train_intersect.values)
            y_train_scaled = self.y_scaler.transform(y_train_intersect.values) 
            
        #intersect and scale the validation data    
        x_val_intersect, y_val_intersect=prepdata.intersect_df(x_val_df, 
                                                               y_val_df)   
                
        
        x_val_scaled = self.x_scaler.transform(x_val_intersect.values)
        y_val_scaled = self.y_scaler.transform(y_val_intersect.values) 
               
        
        self.model.compile(optimizer= Adam(learning_rate=learning_rate), 
                           loss = "mean_squared_error" )
        
        history = self.model.fit( x=x_train_scaled,    
                                  y=y_train_scaled, 
                                  epochs= epochs,
                                  batch_size= batch_size, 
                                  validation_data=(x_val_scaled, y_val_scaled),
                                  verbose= verbose)
        self.histories = self.histories + [history]     
        
 
        
    def plot_history(self):
        """
        plots the loss or accuracy metric and the correspoonding validation 
        metric as function of the epochs, seperate training sessions 
        are shown in different colours
        


        Returns
        -------
        None.

        """
        metric     = 'loss'
        val_metric = f'val_{metric}'
        
        if len(self.histories) > 0:
            loss_fig,loss_ax = plt.subplots()
            
            cumm_epochs = 0
            
            
        
            for i, history in enumerate(self.histories):
                epochs = cumm_epochs + np.arange(
                    len(history.history[metric]))+1
                cumm_epochs= epochs[-1]
                plt.plot(epochs, history.history[metric], 
                         label=f'{metric}, training {i}')
                
                plt.plot(epochs, history.history[val_metric], 
                         label=f'{val_metric}, training {i}')
                
                
                
            plt.legend(bbox_to_anchor=(1.03, 1.0), loc='upper left')
            plt.tight_layout()
            plt.show()   
                
        else:
            print('no training history to plot yet')
            
        
        
        
    def apply(self, application_x):
        """
        Apply the trained model to the forecasted independent parameters
        and make a forecast for the dependent variable    y
        
        Args:
        -------
        forecasted_x, pandas dataframe
        dataframe with the forecasted x_data

        Returns
        -------
        y_forecast, pandas dataframe with datetime index

        """
        # check if the model has been trained
        if len(self.histories) == 0:
            raise Exception('Model has not been trained yet')
            
        # check if the given forecasted varaibles are the same as the
        # ones with which the model was trained with.
        
        if not ( list(self.input_params) == list(application_x.columns) ):
            raise Exception('The x parameters in the aplication data set'
                            '  differ from the ones in the training data set')
        
        
            
        #apply scaling
        ## Apply the same scaling to the test data
        x_app_scaled = self.x_scaler.transform(application_x.values)
                      
        y_forecasted=self.y_scaler.inverse_transform( 
                                             self.model.predict(x_app_scaled) )
        
        y_forecasted_df=pd.DataFrame(y_forecasted, index=application_x.index )
        
        return y_forecasted_df
                                         
        
        
    def save_model(self, filename="../model/NNmodel.pk", notes=""):
        """
        saves the model, together with the metadata needed
        to apply the model to new data,  as pickle file
        
        Keyword args:
        -------------
        filename, string, default: "../model/NNmodel.pk"
            path of the save file
        notes, string, default: ""
            Notes that you might want to add to the model for documentation
            pruposes.
        
        """
        
        model_dict = { 'column_names': self.x_train_df.columns,
                       'model'     : self.model,
                       'trained'   : self.model_trained, 
                       'x_scaler'  : self.x_scaler,
                       'y_scaler'  : self.y_scaler,
                       'history'   : self.history,
                       'notes'     : notes
                      }
        




                
        
    

      
                    