# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:03:33 2024

@author: Jethro Betcke

Module to evaluate model timeseries compared to reference timeseries
it can calulate some accuracy metrics, make plots and create a
mark down report.


"""

__author__     = "Jethro Betcke"
__copyright__  = "Copyright 2024, Jethro Betcke"
__version__    = "0.01"
__maintainer__ = "Jethro Betcke"


# TODO: 
# - force everything into dataframe for consistency    
# - accuracy metrics relative to mean actual       
# - method: figure with histograms of both datasets
# - method: figure with histogram of errors
# - timeseries plot of most accurate week
# - timeseries plot of least accurate week
# - make md report
# - rmse per week or month, per weekday
# nice to have: method get_time_step    
# new class Eval_multi, that compares the results of different methods.


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import prepdata


class Eval:
    
    def __init__(self,y_actual, y_model, 
                 unit='auto', model_name='model', location=''):
        """
        

        Args:
        ---------               
        y_actual: pandas dataseries or dataframe with datetime index                  
            contains the reference data with which the model ouput is compared
            In case of a dataframe with more than one column, only the first
            column will be considered
        y model:   pandas dataseries or dataframe with datetime index  
            contains the model output data that is to be evaluated 
            In case of a dataframe with more than one column, only the first
            column will be considered    
            
        Keyword Args:
        -------------
        unit: string, default: 'auto'
            unit of the data, if set to 'auto', the unit will be searched in 
            the dataframe column header or dataseries title
        modelname: string, default 'model'
            will be used in figures and the mark down report
        location: string, default ''    
            will be used in figures and the mark down report            

        Returns
        -------
        None.

        """

        self.y_actual_cleaned = None
        self.y_model_cleaned = None
        self.unit = unit
        self.model_name = model_name
        self.location = location
                          
        self.data_metrics = pd.DataFrame(columns=['value', 'unit',
                                                 'description']) 
        self.acc_metrics  = pd.DataFrame(columns=['value', 'unit',
                                                 'description'])                 
        
        self.prep_data(y_actual, y_model)
        if unit == 'auto':
            self.get_unit(y_actual)

        
      
        
 

    def prep_data(self,y_actual, y_model):
        """
        Prepares the data for evaluation. It selects the common period of the
        actual and model data and removes data pairs where one of the two data
        points is invalid.
        
        It determines some statistics on the data points
        

        Args:
        ---------               
        y_actual: pandas dataseries or dataframe with datetime index                  
            contains the reference data with which the model ouput is compared
            In case of a dataframe with more than one column, only the first
            column will be considered
        y model:   pandas dataseries or dataframe with datetime index  
            contains the model output data that is to be evaluated 
            In case of a dataframe with more than one column, only the first
            column will be considered  

        Works on Attributes:
        -------
        data_metrics: pandas dataframe with columns: 
                         'value', 'unit','description'
            describes the data in regards of number of points, overlap
            between the datasets, nr of valid datapairs etc.                 
            
        y_actual_cleaned: pandas dataseries with datetime index
        y_model_cleaned :pandas dataseries with datetime index
        unit
        """
        
        #TODO: data_completeness as percentage

        #make sure both inputs are dataseries.
        self.y_actual_cleaned = prepdata.force_series(y_actual)
        self.y_model_cleaned  = prepdata.force_series(y_model)
                               
        #make sure x and y data cover the same period
        intersect_index = self.y_model_cleaned.index.intersection(
                                                   self.y_actual_cleaned.index)
        self.y_actual_cleaned  = self.y_actual_cleaned[intersect_index] 
        self.y_model_cleaned   = self.y_model_cleaned[intersect_index]
                

        #determine the number of data point in the intersection
        self.data_metrics.loc['total nr data pairs','value'] = \
                                                           intersect_index.size
        self.data_metrics.loc['total nr data pairs','unit']  = '1'
        self.data_metrics.loc['total nr data pairs','description'] = \
                                                  'total number of data pairs'
                                                  
        #select the valid data
        valid_actual_mask = np.isfinite(y_actual.values).flatten()       
        self.data_metrics.loc['nr of valid actual points']={}
        self.data_metrics.loc['nr of valid actual points','value'] = np.sum(
                                                             valid_actual_mask)
        self.data_metrics.loc['nr of valid actual points','unit']  = '1'
        self.data_metrics.loc['nr of valid actual points','description'] = (
                  'Number of points of the actual (i.e. reference) data in the'
                                      'intersection, that have a finite value') 
           
        valid_model_mask  = np.isfinite(y_model.values).flatten() 
        self.data_metrics.loc['nr of valid model points','value'] = np.sum(
                                                              valid_model_mask)
        self.data_metrics.loc['nr of valid model points','unit']  = '1'
        self.data_metrics.loc['nr of valid model points','description'] = \
           ('Number of data points of the model in the intersection,'
            ' that have a finite value')        
                                          
        valid_data_pairs_mask =  valid_model_mask & valid_actual_mask
        
        
        #determine the number of data point in the intersection
        self.data_metrics.loc['nr of valid data pairs','value'] = np.sum(
                                                         valid_data_pairs_mask)
        self.data_metrics.loc['nr of valid data pairs','unit']  = '1'
        self.data_metrics.loc['nr of valid data pairs','description'] = \
                             ' number of valid data pairs in the common period'
                             
                             
        self.y_actual_cleaned  = self.y_actual_cleaned.loc[
                                                         valid_data_pairs_mask]
        self.y_model_cleaned  = self.y_model_cleaned.loc[valid_data_pairs_mask]

        
        # for later use in figures and report
        self.period_string = ( f'{self.y_model_cleaned.index[0].date()}'
                               f' to {self.y_model_cleaned.index[-1].date()}' )
        
        # the maximum of 
        self.max_of_max= np.max( [self.y_model_cleaned.max(),
                                  self.y_actual_cleaned.max()] )
        
        self.min_of_min= np.max( [self.y_model_cleaned.min(),
                                  self.y_actual_cleaned.min()] )
           
           
                      

    def get_unit(self,y_actual) :
        
        """
        searches the headers of y_actual aand y_model for text between brackets,
        which is interpreted as the unit of the data
        
        Uses Attributes:
        ----------------
        y_actual_cleaned
        y_model_cleaned
        
        Works on Attributes:
        ----------
        self.unit
            
        """
        
        self.unit = 'unknown'
        var_headers = [self.y_actual_cleaned.name, self.y_model_cleaned.name]
        
        for var_header in var_headers:
            #search for text between brackets
            units = re.findall(r'\(.*?\)', var_header)
        
        if len(units) > 0:
           self.unit = units[-1] 
           return
       
        return
    

    
   
    def calc_acc_metrics(self):    
        

        """
        calculates several metrics of accuracy and stores the results in
        a pandas dataframe
        
        Uses Attributes:
        ---------               
        y_actual_cleaned
        y_model_cleaned     
        
        
        Works on Attributes:
        -----------
        acc_metrics
        
                
        """
        
        #first some metrics for the actual and model data seperately
        self.acc_metrics.loc['Mean Actual','value'] = \
                                                   self.y_actual_cleaned.mean()
        self.acc_metrics.loc['Mean Actual','unit'] = self.unit                                           
        self.acc_metrics.loc['Mean Actual','description'] = ('Arithmic Average'
                                                       'of the reference data')

        #first some metrics for the actual and model data seperately
        self.acc_metrics.loc['Stdv. Actual','value'] = \
                                                   self.y_actual_cleaned.std(
                                                                       ddof=0)
        self.acc_metrics.loc['Stdv. Actual','unit'] = self.unit                                           
        self.acc_metrics.loc['Stdv. Actual','description'] = ('standard'
                      ' deviation of the reference data, normalised with N')


                                           
                                                   
        self.acc_metrics.loc['Mean Model','value'] = \
                                                   self.y_model_cleaned.mean()
        self.acc_metrics.loc['Mean Model','unit'] = self.unit                                           
        self.acc_metrics.loc['Mean Model','description'] = ('Arithmic Average'
                                                           'of the model data')


        self.acc_metrics.loc['Stdv. Model','value'] = \
                                              self.y_model_cleaned.std(ddof=0)
        self.acc_metrics.loc['Stdv. Model','unit'] = self.unit                                           
        self.acc_metrics.loc['Stdv. Model','description'] = ('standard'
                         ' deviation of the model data, normalised with N')
                            
        
        
        #Comparison metrics
        # RMSE and its components (RMSE^2 = MBE^2 + BoSD^2  +Disp^2)
        self.acc_metrics.loc['RMSE','value'] = np.sqrt(
                                          metrics.mean_squared_error(
                                                      self.y_actual_cleaned, 
                                                      self.y_model_cleaned ) )
        self.acc_metrics.loc['RMSE','unit']  = self.unit
        self.acc_metrics.loc['RMSE','description']  = ('Root Mean Square Error'
                                    'Note: RMSE^2= MBE^2 + BoSD^2 +disp^2')  
        

        self.acc_metrics.loc['MBE','value'] = (
                                  self.acc_metrics.loc['Mean Model','value']
                                - self.acc_metrics.loc['Mean Actual','value'] )
        self.acc_metrics.loc['MBE','unit']  = self.unit
        self.acc_metrics.loc['MBE','description'] = 'Mean Bias Error'
        

        self.acc_metrics.loc['BoSD','value'] = (
                                 self.acc_metrics.loc['Stdv. Model','value'] 
                               - self.acc_metrics.loc['Stdv. Actual','value'] )
        self.acc_metrics.loc['BoSD','unit'] = self.unit
        self.acc_metrics.loc['BoSD','description'] = ('Bias of stdv = '
                                               'stdv(method)-stdv(reference)')
        
        self.acc_metrics.loc['Disp','value'] = np.sqrt(
                                      self.acc_metrics.loc['RMSE','value']**2 
                                    - self.acc_metrics.loc['MBE','value']**2
                                    - self.acc_metrics.loc['BoSD','value']**2)
        self.acc_metrics.loc['Disp','unit'] = self.unit
        self.acc_metrics.loc['Disp','description']=\
             'sqrt{Dispersion according to 2*[1-corr(method,ref)](stdv(method) stdv(ref) }'        
               
        
        #other metrics

        self.acc_metrics.loc['MAE','value'] = metrics.mean_absolute_error(
                                                         self.y_actual_cleaned, 
                                                         self.y_model_cleaned)
        self.acc_metrics.loc['MAE','unit']  = self.unit
        self.acc_metrics.loc['MAE','description']  = 'Mean Absolute Error'
        
        
        
        
        
        self.acc_metrics.loc['Pearson corr.','value'] = \
            self.y_actual_cleaned.corr( self.y_model_cleaned, method='pearson')

        self.acc_metrics.loc['Pearson corr.','unit'] = '1'
        self.acc_metrics.loc['Pearson corr.','description'] = \
                                              'Pearson correlation coefficient'
        
        
        
        self.acc_metrics.loc['R2_score','value'] = metrics.r2_score(
                                                         self.y_actual_cleaned, 
                                                         self.y_model_cleaned)

        self.acc_metrics.loc['R2_score','unit'] = '1'
        self.acc_metrics.loc['R2_score','description'] = ('sklearn R^2 score'
                                                          'for correlation')
        
        
        
    def scatter_plot(self):
        """
        creates a straightforward scatter plot of the model results
        against 
        
        Uses Attributes:
        ---------               
        y_actual_cleaned
        y_model_cleaned 
        
        Creates/Overwrites Attributes:
        ------------------------------
        scat_fig: figure handle
        scat_ax : axes handle

        Returns
        -------
        None.

        """
        self.scat_fig,self.scat_ax = plt.subplots()
        
        #self.scat_ax.set_aspect('equal', adjustable='box')
        self.scat_ax.set_title(f'{self.location}  {self.period_string}')
        self.scat_ax.set_xlabel(f'Actual ({self.unit})')
        self.scat_ax.set_ylabel ( f'{self.model_name} ({self.unit})')
        
        self.scat_ax.plot(self.y_actual_cleaned.values, self.y_model_cleaned.values, 'k.')
        
        #make the plot square
        
        self.scat_ax.set(adjustable='box',aspect='equal')

        xlim = self.scat_ax.get_xlim()
        ylim = self.scat_ax.get_ylim()
        
        newlim = [0,1]
        newlim[0]=np.min([xlim[0],ylim[0]])
        newlim[1]=np.max([xlim[1],ylim[1]])
        
        self.scat_ax.set_xlim(newlim)
        self.scat_ax.set_ylim(newlim)    
        
        self.scat_ax.plot(newlim,newlim, 'g--')
        plt.tight_layout()        
        plt.show()
        
    
    def timeseries_plot(self):
        """
        Creates a timeseries plot of y actual and y forecast.
        
        Uses Attributes:
        ---------               
        y_actual_cleaned
        y_model_cleaned 
        model_name
        
        Creates/Overwrites Attributes:
        ------------------------------
        ts_fig: figure handle
        ts_ax : axes handle

        Returns
        -------
        None.

        """
        self.ts_fig,self.ts_ax = plt.subplots()  
        self.ts_ax.set_title(f'{self.location}  {self.period_string}')
        self.ts_ax.set_xlabel('date time')
        self.ts_ax.set_ylabel ( f'Power Demand ({self.unit})')
        plt.xticks(rotation=-45) 
        
        self.ts_ax.plot(self.y_actual_cleaned.index,
                        self.y_actual_cleaned.values, 'k-',
                        label='actual')
        self.ts_ax.plot(self.y_model_cleaned.index,
                        self.y_model_cleaned.values, 'r:d',
                        label=self.model_name)
        
        plt.legend(bbox_to_anchor = (0.5, 1.1), loc='lower center')
            
        plt.tight_layout()
        plt.show()
        

    def hist_compare_plot(self) :
        """
        Creates a plot with the histograms of y_actual and y_forecast

        Uses Attributes:
        ---------               
        y_actual_cleaned
        y_model_cleaned 
        min_of_min
        max_of_max
        

        Creates/Overwrites Attributes:
        ------------------------------
        hist_fig: figure handle
        hist_ax : axes handle
            

        Returns
        -------
        None.

        """
        
        #make sure both histograms use the same bins
        nrofpoints=self.data_metrics.loc['nr of valid actual points','value']        
        if (nrofpoints<500):
            nrofbins=10
        else:
            nrofbins=100
            
            
        total_spread = self.max_of_max - self.min_of_min  
        binwidth = total_spread / nrofbins
            
        bins= np.arange(self.min_of_min,self.max_of_max + binwidth, binwidth)   
        
        #define labels
        self.hist_fig,self.hist_ax = plt.subplots()
        self.hist_ax.set_title(f'{self.location}  {self.period_string}')
        self.hist_ax.set_ylabel ( 'nr. of points per bin')
        self.hist_ax.set_xlabel ( f'Power ({self.unit})')

        self.hist_ax.hist(self.y_actual_cleaned,bins=bins,alpha=0.5, color='blue',
                     label='actual ')
        self.hist_ax.hist(self.y_model_cleaned,bins=bins,alpha=0.5, color='orange',
                     label=self.model_name)
        plt.legend(bbox_to_anchor = (0.5, 1.1), loc='lower center')
        
        
        plt.tight_layout()
        plt.show()


        






        
        
        