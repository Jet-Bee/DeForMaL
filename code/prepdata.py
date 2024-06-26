# -*- coding: utf-8 -*-
"""

Collection of tools to prepare input data for the prognosis of power demand
for small European countries.

The main principle is that all input data are created as pandas dataframes and
seperately saved as csv files. The underlying concept is that this enables
the creation of different combinations of variables that can be used to train
and test a machine learning model for power demand forecasting. 

The dataframes have the following properties:
    - Their index is a datetimeindex in local time
    - resolution is an hour
    - Where applicable the data is averaged over this hour
    - The timestamp indicates the start of the averaging period. This follows
      the convention of the entsoe power data
    - Columns have a unique name. This makes it possible to check if the input 
      variables for the application of the model are the same as the ones used
      for the training of the model.
    - categorical variables  
      

Functions to derive time variables from a datetimeindex:  
    - hour_variable(): hour of the day as continuous or categorical variable
    - type_day()     : day of the week, with optionally midweek days combined
    - daylightsaving_categories: wintertime, summertime and switchdays
    
Classes to pull data from APIs:
    - EntsoePower: get and process power demand data from ENTSO-E
    - OpenHolidays : get and process holidays from openholidaysapi.org
    
Helper functions and utilities:
    - dictionary:  coupling country codes with timezone names
    - one_hot_df: apply one hot encoding to categorical value in pandas 
      dataframe
    - combines_dfs: combine several timeseries dataframes into a single new one
    - read_login:   read the login data for the different APIs stored in a
                    textfile
    
        



collections of tools to pull data relevant to the DePro
software from several sources from the internet.

Power demand: https://transparency.entsoe.eu

holidays:https://www.openholidaysapi.org/en/
"""

__author__     = "Jethro Betcke"
__copyright__  = "Copyright 2024, Jethro Betcke"
__version__    = "0.02"
__maintainer__ = "Jethro Betcke"


# TODO add class to pull weather data (openweather/ECMWF/ERA5)
# TODO holidays: select only national holidays, or make an option for this
# TODO make structure and names of the different classes more similar
#
# DONE: workaround to deal with the maximum of 3 years of data 
#                that can be pulled from openholidaysapi.org
# DONE: bridgedays: create category for days between holiday and weekend

import os
from os.path import splitext
import requests
import cdsapi
from entsoe import EntsoePandasClient
import pandas as pd
import numpy as np
import xarray as xr
import time
import pickle
import shutil








#coupling ISO country codes with python timezones
#https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
# import pytz; pytz.all_timezones

country_timezone={'AD':'Europe/Andorra',
                  'CY':'Europe/Nicosia',
                  'IM':'Europe/Isle_of_Man',
                  'JE':'Europe/Jersey',
                  'LU':'Europe/Luxembourg',
                  'LI':'Europe/Vaduz',
                  'MK':'Europe/Skopje'}


# a square approximation of the country
# with [northside, westside, southside, eastside]
country_area={'AD':[42.6,1.45,42.45,1.75],
              'CY':[35.35,32.35,34.85,34.05 ],
              'IM':[54.40,-4.75, 54.05,-4.35],
              'JE':[49.25,-2.25,49.15,-2.05],
              'LU':[50, 5.75, 49.5, 6.3],
              'LI':[47.25,9.50,47.05,9.60],
              'MK':[42.20,20.55,41.10,22.95 ]
             }



# HELPER FUNCTIONS AND UTILITIES


def one_hot_df(categorical_df, column_prefix=''):
    """
    Applies one hot encoding to a pandas dataframe.
    The motivation no to use keras 'to_categorical()' is that we want to 
    preserve the category names. This makes it easier to reuse the model
    
    Args:
    ------------
    categorical_df, pandas dataseries or pandas dataframe with one column
        contains categorical data
        
    Returns:
    ------------
    one_hot_enc_df, pandas dataframe
        contains the input data in one hot encoded form,
        i.e. when there are N categories there are N-1 columns
        each column has a value of one where the category applies
        and is zero where it does not apply. Each column has the name
        of the category as found in timeseries_df
    
    
    """
    
    if categorical_df.shape[1] != 1:
        raise Exception('can one deal with one column input (for now)')
    else:
        # with get_dummies each value in the dataset gets a column,
        # with a boolean value to indicate if that value occurs
        # get_dummies only works on dataseries, so the df must be converted
        # using squeeze
        one_hot_encoded_df = pd.get_dummies(categorical_df.squeeze()) 

        #remove redundant last column and convert boolean to integer
        one_hot_encoded_df = ( one_hot_encoded_df
                               .drop(one_hot_encoded_df.columns[-1] ,axis=1)
                               .astype(int) )
   
        # add a prefix to the column names if desired
        if column_prefix != '':
            old_column_names = one_hot_encoded_df.columns
            one_hot_encoded_df.columns = [ f'{column_prefix}{col}' 
                                                  for col in old_column_names ]
        
        return one_hot_encoded_df  
    
    
    
def combines_dfs(*argv):
    """
    combines the timeseries dataframes into a single dataframe
    It checks if the datetimeindex of each frame is in the same time system
    and takes the temporal intersection if the timeperiods are not completely 
    the same.
    
    Args:
    -------------    
    *argv timeseries dataframes that are to be combined,
    either as seperate inputs or as a list
    
    Returns:
        
    result_ts_df    
    
    """
    
    if len(argv)==1 and type(argv[0])==list:
        df_list=argv[0]
    else:
        df_list=argv
        
    result_ts_df=df_list[0]    
    
    for i,df in enumerate(df_list[1:]):
        #check if in same timezone
        if result_ts_df.index.tz==df.index.tz:
           result_ts_df=pd.merge(result_ts_df, df, how='inner',
                                 left_index=True, right_index=True)
        else  :
            raise Exception(f'input {i} is not in the same timezone ',
                            'as previous inputs')   
    return result_ts_df    


def force_df(in_data):
    """
    checks if an input is a pandas series or pandas dataframe
    if it is a series the input will be converted to dataframe.
    if it is already a dataframe the output will equal the input
    
    throws an exception for any other input type
    
    Args:
    -------    
    in_data: pandas dataframe or series.
    
    Returns
    -------
    out_data: pandas dataframe
        contains the same data as in_data

    """
    
    in_type=str(type(in_data)).strip('()<>\'"')
        
    if in_type.endswith('Series'):    
        out_data = in_data.to_frame()
    elif in_type.endswith('DataFrame'):
        out_data = in_data  
    else:
        raise Exception('input can only be pandas DataFrame or'
                            ' pandas Series')
    return out_data   
        
        
def force_series(in_data):
    """
    Checks if an input is a pandas series or pandas dataframe
    if it is a dataframe the input will be converted to a series.
    If there is more than one column in the dataframe, only the first
    column will be used for the series
    
    If it is already a dseries the output will equal the input
    
    throws an exception for any other input type
    
    Args:
    -------    
    in_data: pandas dataframe or series.
    
    Returns
    -------
    out_data: pandas dataframe
        contains the same data as in_data

    """     

    in_type=str(type(in_data)).strip('()<>\'"')        
    
    if in_type.endswith('DataFrame'):    
        out_data = in_data.iloc[:,0].squeeze()
    elif in_type.endswith('Series'):
        out_data = in_data  
    else:
        raise Exception('input can only be pandas DataFrame or'
                            ' pandas Series')
    return out_data         

    

def split_df_on_date(splitdate, x_data_df ):
    """
    
    Splits a dataframe with a datetime index on a given date.
    The timepoint 00:00 of the splitdate will be the first timepoint of
    the second dataframe(s). 


    Args
    ----------
    splitdate : string
        Date in format 'YYYY-MM-DD'
    
    x_data_df : pandas dataframe with datetime index
        

    Returns
    -------
    x_before_df : pandas dataframe with datetime index
        first part of x_data_df up to splitdate 0:00
        (datetime index < splitdate 0:00)
    
    x_after_df : pandas dataframe with datetime index
        second part of x_data_df starting at splitdate 0:00
        (datetime index >= splitdate 0:00)
    """
    #TODO: also accepte datenum or pandas timestamp
    
    
    
    train_bool_index = x_data_df.index.date < pd.Timestamp(splitdate).date()           
    
    x_before_df = x_data_df[train_bool_index]
    x_after_df  = x_data_df[~train_bool_index]
    
    return x_before_df, x_after_df


def intersect_df(in_df1, in_df2):
    """
    Determines the common datetime points of two dataframes (or pandas series)
    that each have a datetime index, and gives back the dataframes for these 
    common datapoints. The outputs are always data frames.
    
    Args
    -------
    in_df1: pandas dataframe, or series, with datetime index
    in_df2: pandas dataframe, or series, with datetime index
       

    Returns
    -------
    out_df1: pandas dataframe with datetime index
        in_df1 only containing the rows correponding with datetime points
        that are also present in in_df2
    out_df2: pandas dataframe with datetime index
        in_df2 only containing the rows correponding with datetime points
        that are also present in in_df1

    """
    
    # make sure x and y data cover same period       
    intersect_index = force_df(in_df1).index.intersection(
                                                        in_df2.index)
    
    if intersect_index.size == 0:
        raise Warning('No common datetime points between inputs')
        
    out_df1  = force_df(in_df1).loc[intersect_index,:]
    out_df2  = force_df(in_df2).loc[intersect_index,:]    

    return out_df1, out_df2
        


def read_login(loginfile, datasource):
    """
    Reads the username and password from the csv login file.
    
    the loginfile has the lines:
    '<datasource>  <username> <password>'
    
    Args:
    ----------
    loginfile:string 
    text file where login tokens are stored.
    the file has to contain the lines:
    '<datasource> <security_token>'  

    
    Returns: 
    ----------
    token
             
    """
    
    with open(loginfile,'r') as lofi:
        lines=lofi.readlines()

        for line in lines:
            lineparts=line.split()
                            
            if ( len(lineparts)>0) and \
                (lineparts[0].strip().lower()==datasource):
                #TODO check length len(lineparts>2)
                if len(lineparts)<2:
                    raise Exception(f'logindata for {datasource} missing')
                else:    
                    token=lineparts[1]
                    if token.strip()=='please_fill_in':
                        raise Exception('fill in your personal token for',
                                        f' {datasource} into {loginfile}')                        
                    
                    return token
            
        #No logindata could be retrieved so raise Exception
    raise Exception(f'No logindata for data source {datasource}',
                         'in file {loginfile}')    
    
    
        
# FUNCTIONS TO DERIVE TIME VARIABLES
# These function require always a datetime index as input 
# and always return a dataframe that have that datetime index as index.

def hour_variable(dt_index, as_category=True):
    """
    creates a timeseries dataframe with either one column with the hour,
    or a one hot coded 
    

    Args
    ----------
    dt_index : pandas datetime index
        
        
    Keyword Args     
    -----------
    as_category : Boolean, default is True
        if True: the hours 0:23 will be treated as categories,
                 and one hot encoded
        if False: the hours will be treated as a semi-continuous variable,
                  and the returned dataframe will have only one column, with 
                  the hour as integer.

    Returns
    -------
    hour_df pandas dataframe
        dataframe with dt_index as index, see the description of the 
        as_category keyword for details on the columns.

    """
    
    hour_df = pd.DataFrame(dt_index.hour,index=dt_index, columns=['hour']) 
    
    if as_category:
       hour_df =  one_hot_df(hour_df, column_prefix='hour_')
    
    return hour_df             
    


def type_day(dt_index, combine_midweekdays=False): 
    """
    create one-hot encoded dataframe of typedays
    
    Args:
    -------------
    timeseries_df pandas dataseries or dataframe
        must contain a datetime index
        the datetime index will be used to determine the typedays
        
    
    Key Args:
    -------------
    combine_midweekday Boolean, default False
        if False every day of the week will be seen as a different typeday,
            i.e. there will be seven typedays.
        if True Tuesday, Wednesday and Thursday will be categorised as
        the same typday, i.e. there will be five typedays
        
    Returns:
    ------------    
    typeday_ohe_df pandas dataframe
        one hot encoded typedays, column names will be of the form
        'typeday_<N>'
        
    """       
    #dayofweek is given out as integers 0 to 6 with Monday as 0
    typeday_df = pd.DataFrame(dt_index.dayofweek,index=dt_index, 
                              columns=['day of week'])
        
    if combine_midweekdays:
        midweek_mask = (typeday_df.values>0) & (typeday_df.values<4)
        typeday_df.loc[midweek_mask]=2
         
        
    typeday_ohe_df = one_hot_df(typeday_df,column_prefix='typeday_')    
        
    return  typeday_ohe_df



def daylightsaving_flags(dt_index, switchdaylag=0) :
    """
    creates a binary dataframe with the following categories:
    wintertime, summertime, switchday
    where switchday is the day when the clock is shifted, and optionally some
    "lag" days where the effect of this change is still noticable.
    There is no need for a seperate label for the lag days, because these
    will be on a different day of the week than the "true" switch day.
    
    There is also no need for a seperate label for the spring and autumn switch day
    because the combination with the summertime label already gives this 
    information.
    
   
    Args:
    ------------
    dt_index: pandas date time index
    The timesystem must be in the local time, including daylight savings time
    It must have a resolution of one hour.
        
    Keyword Args:
    -------------   
    switchdaylag, int, default 0
        additional days that count as switch day 
    
    """
    
    #create output   
    dls_df=pd.DataFrame(index=dt_index, 
                        columns=['summertime',
                                 f'switchday_lag_{switchdaylag}'])
    
    #create binary summertime indicator
    
    summertime = dt_index.map( lambda dt: int( dt.dst()/ pd.Timedelta('1h')) )
    
    dls_df.loc[:,'summertime'] = summertime
    
    summertime_np = np.asarray(summertime)
    
    #now find the switch days
    switch_point_index = np.where(summertime_np[0:-1]!=summertime_np[1:] )
    ##get the dates of the switches
    switchdates=dt_index[switch_point_index].date
    
    ## now mark  all switchdays and the following lagdays.
    
    dates=dt_index.date
    
    switchday_flag = np.zeros(len(dates))
    
    for switchday in switchdates:
        for lagday in range(switchdaylag+1):
            delta_day = lagday*pd.Timedelta('1day')
            lagged_date = switchday+delta_day                        
            switchday_flag[dates == lagged_date] = 1
            
    
    dls_df.loc[:,f'switchday_lag_{switchdaylag}'] = switchday_flag.astype(int)
        
    
    return dls_df



# CLASSES TO PULL DATA FROM APIs
           
    
class EntsoePower:
    
    """
    class to pull and process power demand data from the API provided by
    European Network of Transmission System Operators for Electricity
    (ENTSO-E: https://transparency.entsoe.eu)
    
    Methods:
    -------------     
    __init__: 
        defines an EntsoePower object
    pull_process_save: 
        pulls actual and forecasted power demanddata from 
        the EntSoe server, averages it to hourly data and saves forecast and 
        actual data as seperate pickled dataframes
        
    pull_all: 
        pulls actual and forecaste powerdemanddata from the EntSoe server
    to_hourly: 
        converts the original 15 minute data to hourly data
    split:
        split the actual and forecasted data in seperate dataframes.
   
    """
    
    def __init__(self, country_code, startdate, enddate, 
                 loginfile='../userdata/logins.txt'):
        """
        Args
        ----------
        datatype: string
            
        startdate : string 
            date of the first day of the period to be pulled
            format: 'YYYY-MM-DD'
    
        enddatetime : string 
            date of the last date of the period to be pulled
            format: 'YYYY-MM-DD'
        country :string
           two letter iso country code.
        loginfile : string
            file where the entsoe username and password are stored.
            the file has to contain the line:
            'entsoe  <username>  <password>'    
        
        Keyword Args:
            passwordfile: string, default '../userdata/logins.txt'
                    
        """  

        self.startdatetime = f'{startdate}T00:00'
        self.enddatetime   = f'{enddate}T23:45'
        self.country_code  = country_code     
        self.loginfile     = loginfile
        self.entsoe_key    = read_login(self.loginfile, 'entsoe')
        self.time_zone     = country_timezone[self.country_code]
        
    def pull_process_save(self,filename):
        """
        one stop shop to:
            - pull the 15 minute data from the API
            - average it to 1h resolution
            - split the data in forecast and actual
            - save the results to csv
            
        File Output:
        ------------
        <filename>: 
            csv file with hourly forecasted and actual load
        <filenamebase>_forecast<extension>: 
            csv file with only the hourly forecasted load
        <filenamebase>_actual<extension>: 
            csv file with only the hourly actual load                
                                           
        
        """
        self.pull_all()
        self.to_hourly()
        self.data_hr.to_pickle(filename)
           
        self.split()
        
        filebase,extension = os.path.splitext(filename)
        self.forecasted_load_hr.to_pickle(f'{filebase}_forecast{extension}')
        self.actual_load_hr.to_pickle(f'{filebase}_actual{extension}')  
            
                          
    
    def pull_all(self):
        """
        pull the actual power demand and the day ahead power forecast from the
        entsoe API, both have an original resolution of 15 minutes
        
        The timesytem is set to the local time system.
        
        Uses Attributes:
        -------
        startdatetime, enddatetime, time_zone, entsoe_key 
           
        Creates Attribute:
        --------    
        data_raw: pandas dataframe
           dateframe with datetime index in local time, with 15 minute 
           resolution, and columns 'Forecasted_Load' and 'Actual_Load' (MW)
           Load data is averaged over the 15 minute period
           the timestamp indicates the start of the averaging interval

        """
        
        start = pd.Timestamp(self.startdatetime, tz=self.time_zone)
        end = pd.Timestamp(self.enddatetime, tz=self.time_zone)
        entsoe_client = EntsoePandasClient(api_key=self.entsoe_key)
        
         
        self.data_raw = entsoe_client.query_load_and_forecast(
                                           self.country_code, 
                                           start=start, end=end)
        
        #replace spaces in column headers with underscores
        self.data_raw.columns = [col_head.replace(' ', '_') 
                                  for col_head in self.data_raw.columns]
        
        
    def to_hourly(self) :
        """
        Averages the raw data 15 minute data to hourly data, the new
        timestamp indicates the start of the hour

        Returns
        -------
        None.
        
        Uses Attributes:
        -------    
        data_raw, pandas dataframe
            original 15 minute load data, see pull_all()
        
        Creates Attribute:  
        -------    
        data_hr, pandas dataframe
           dateframe with datetime index in local time, with one hour 
           resolution, and columns 'Forecasted_Load' and 'Actual_Load' (MW)
           Load data is averaged over the 60 minute interval
           the timestamp indicates the start of the averaging interval

        """
        
        self.data_hr = self.data_raw.resample('h', origin='start_day').mean()
        
        
    def split(self):
        """
        
        splits the hourly data in two dataframes
        one for the forecast, and one for the actual load values. 
        
        The forecast can possible be used as additional input into the
        Machine learning model, so it must be seperated from the actual
        values.

        Returns
        -------
        None.
        
        Uses Attributes:
        --------   
        data_hr, pandas dataframe
            hourly load data, see to_hourly()       
        
        Creates Attributes:
        --------    
        forecasted_load_hr, pandas dataframe
            hourly forecasted load
        actual_load_hr
            hourly actual load

        """
        
        self.forecasted_load_hr = self.data_hr['Forecasted_Load']
        self.actual_load_hr     = self.data_hr['Actual_Load']
                           
                               
                

#def add_summertime(df_in)         :
    

class OpenHolidays:
    
    """
    Class to get information on holidays from openholidaysapi.org
    
    Note: The data starts at 2020. However, when you try to pull more than
    three years of data, an error code 500 is given out by the API, and no
    data is returned
    
    """

    def __init__(self,country_code,startdate,enddate):       
        """
        Args
        ----------
        country code: string
            two letter ISO639 country code
        startdate, string
            first date of the period to be pulled  in format "YYYY-MM-DD"  
        enddate, string
            last date of the period to be pulled  in format "YYYY-MM-DD"         
        """    
   
        self.country_code = country_code
        self.time_zone = country_timezone[self.country_code]
        self.startdate_str = startdate
        self.enddate_str   = enddate
        self.startdate = pd.to_datetime(startdate)
        self.enddate   = pd.to_datetime(enddate)
               
        
    def pull_process_save(self,groupmethod, filename='../data/holidays.pkl'):
        """

        one stop shop to:
            - pull the data from the  openholidays API
            - convert the json style data to a dataframe calendar
            - group the holidays by type or name
            - resample the holiday flags to hourly resolution
            - save the result to file
        
        Args
        ----------
        groupmethod, string
            - when value is 'holidaytype' there will be three categories:
              bank holidays, public holidays and school holidays    
            - when value is 'holidayname' each holiday will have its own 
              category
              
        Keyword Args
        -----------
        filename, string, default: '../data/holidays.pkl'
        file in which the results are saved
        
        File Output
        ------------
        a csv file with an hourly date time index in the local time system
        and one column for each holiday category. The value is 1 if the 
        holiday group category applies at that time, else zero.
        
                
        """
        self.pull_all()
        self.json2df( groupmethod=groupmethod)
        self.to_hourly()        
        self.holiday_hr.to_pickle(filename)
                             
    
    def pull_all(self):
        self.public_holidays = self.pull_holidays('Public')
        self.school_holidays = self.pull_holidays('School')        
        self.all_holidays    = self.public_holidays + self.school_holidays
    
    
    def pull_holidays(self,holidaytype):
        """        
        Args
        ----------
        holidaytype : string
            'public' or 'school'.
            
        Returns
        ----------
        response.json(): dict
            the requested holiday data in json format
        """
                      
        startyear = self.startdate.year
        endyear   = self.enddate.year
        
        
        nr_of_years= endyear-startyear+1        
        nr_of_requests=int(np.ceil(nr_of_years/3.0)) #max. 3 years per request
        
        holidays_json =[]
        
        
        for i in range(nr_of_requests):
            if i == 0:
                request_startyear = startyear
                request_startdate_str = self.startdate_str
            else:
                request_startyear = startyear + i*3
                request_startdate_str = f'{request_startyear}-01-01'
                
            if i == (nr_of_requests - 1):
                request_enddate_str = self.enddate_str
            else:
                request_endyear = request_startyear + 2
                request_enddate_str = f'{request_endyear}-12-31'
                       
        
            requestline = (f'https://openholidaysapi.org/{holidaytype}'
                           'Holidays?'
                           f'countryIsoCode={self.country_code}&' 
                           'languageIsoCode=EN&'
                           f'validFrom={request_startdate_str}&'
                           f'validTo={request_enddate_str}')
        
        
            response = requests.get(requestline)
        
            if response.status_code != 200:
                raise Exception('Data request to openholidaysapi.org '
                                f'failed with code {response.status_code}')
        
            holidays_json = holidays_json + response.json()
        
        return holidays_json
    
    
    def extract_holiday_names(self):
        """
        Helper method to the json2pd_fd method

        Returns
        -------
        holiday_name_lst: list
        
        list of the unique names of the holidays,
        sorted alphabetically
        """
    
        raw_name_lst=[holiday['name'][0]['text'] \
                                    for holiday in self.all_holidays ]
        holiday_name_lst=list(set(raw_name_lst))
        holiday_name_lst.sort()
        
        #print(*holiday_name_lst)
        
        return holiday_name_lst
        
        
    def json2df(self,  groupmethod='holidaytype'):
        """
        create a pandas "calender" each row represents a date
        dependent on the groupingmethod each column represents a
        holiday or a group of holidays
        the value in the cells will be 0 (no holiday) or 1 (holiday)   
        
        Keyword  Args
        ---------
        groupmethod, string, default 'holidaytype'
            - when value is 'holidaytype' there will be three categories:
              bank holidays, public holidays and school holidays    
            - when value is 'holidayname' each holiday will have its own 
              category
        """
                       
        start_date4_df = pd.to_datetime(self.startdate_str, yearfirst=True)
        end_date4_df   = pd.to_datetime(self.enddate_str, yearfirst=True)
        
       
        # by setting inclusive to left 0:00 of the next day is excluded
        # form the timeseries
        #dt_index=pd.date_range(start=start_date4_df, end=end_date4_df, 
        #                       freq=resolution, inclusive='left')
        date_index=pd.date_range(start=start_date4_df, end=end_date4_df, 
                                freq='D')
        
        if groupmethod == 'holidaytype':
            column_index  = ['Public','Bank','School']
            holiday_df = pd.DataFrame(0,index=date_index,columns=column_index)
            for holiday in self.all_holidays:
                holiday_start = pd.to_datetime(holiday['startDate'],
                                                              yearfirst=True)
                holiday_end = pd.to_datetime(holiday['endDate'],yearfirst=True)
                holiday_df.loc[holiday_start:holiday_end, holiday['type']] = 1
               
        elif groupmethod == 'holidayname':
            
            column_index = self.extract_holiday_names()
            holiday_df = pd.DataFrame(0,index=date_index,columns=column_index)
            for holiday in self.all_holidays:
                holiday_start = pd.to_datetime(holiday['startDate'],
                                                              yearfirst=True)
                holiday_end = pd.to_datetime(holiday['endDate'],yearfirst=True)
                holidayname = holiday['name'][0]['text']                
                holiday_df.loc[holiday_start:holiday_end, holidayname]=1 
        else:
            raise Exception(f'unknown groupmethod {groupmethod}')
                               
        
        # bring  the dates from tz naive to local timezone
        
        self.holiday_day = holiday_df.tz_localize(self.time_zone)
        
    
    def to_hourly(self):
        """
        resample the daily holiday flags to an hourly resolution.
        
        Returns
        -------
        None
        """
        resolution = 'h'        
        
        #pandas resample does not extrapolate, 
        #so to make sure the last day is completely included in the 
        #timeseries, 0:00 of the next day is added to the dataset
        
        extraday = pd.DataFrame(index = [self.holiday_day.index[-1] 
                                                        + pd.Timedelta('1D')],
                                columns = self.holiday_day.columns )                                          
                
        holiday_day_extended = pd.concat([self.holiday_day,extraday])
        
        #resample to the desired resolution
               
        self.holiday_hr = holiday_day_extended.resample(resolution).ffill()
        
        #remove the last row that was only there to make sure the
        #last day was filled in completely
        self.holiday_hr.drop(index=self.holiday_hr.index[-1],
                            axis=0,inplace=True)
        


def bridgedays(holidaytype_df, xmas2NY_as_bridge=True):
    """
    Determines which days are bridgedays, i.e. the day between a public holiday
    and the weekend. So bridgedays are either a Monday or a Friday. If a 
    bridgeday is also a bankholiday it is not marked as a bridgeday.

    Args.:
    ----------
    holidays_df : pandas dataframe with datetime index
        has at least the column 'Public' 
        where the values can be 0 or 1.
        
    Keyword Args:
    -------------    
        
    xmas2NY_as_bridge : Boolean, default True
        If True the period between Christmas and new year will be
        put in the bridgeday category.

    Returns
    -------
    bridgeday_df: pandas dataframe
        contains the column 'Bridgeday' the value is 1 when it is a bridgeday
        
    """
    
    #get the weekdays (Monday=0, Sunday=6):

    weekday_df = pd.DataFrame(holidaytype_df.index.dayofweek,
                              index=holidaytype_df.index, 
                              columns=['day of week'])
    
    #get the normal bridgedays
    
    monday_bridge_dt = holidaytype_df.index[
                                     (holidaytype_df['Public'].values == 1) 
                                 & (weekday_df['day of week'].values == 2 ) 
                                        ]  - pd.to_timedelta('24h')
    
    friday_bridge_dt = holidaytype_df.index[
                                     (holidaytype_df['Public'].values == 1) 
                                 & (weekday_df['day of week'].values == 4 ) 
                                        ]  + pd.to_timedelta('24h') 
    
    #remove bridgedays outside the original datetime range 
    monday_bridge_dt=holidaytype_df.index.intersection( monday_bridge_dt)
    friday_bridge_dt=holidaytype_df.index.intersection( friday_bridge_dt)
    
    bridgeday_df = pd.DataFrame( index=holidaytype_df.index, 
                                 columns=['Bridgeday'])
    
    bridgeday_df.loc[:,'Bridgeday']= 0
    bridgeday_df.loc[monday_bridge_dt,'Bridgeday'] = 1
    bridgeday_df.loc[friday_bridge_dt,'Bridgeday'] = 1  
    
    #Note: Dec 26th is not a holiday in every country, so the 26th is choosen
    # as the first day of the bridge period between christmas and new Year
    # if it is a holiday it is removed as a bridgeday in the step below.
    if xmas2NY_as_bridge:
       bridge_mask = (bridgeday_df.index.month == 12) \
                    & (bridgeday_df.index.day > 25)    \
                        
       bridgeday_df.loc[bridge_mask,'Bridgeday'] = 1     
       #remove the weekend  in this period
       weekend_mask = (weekday_df['day of week'].values == 5) \
                     | (weekday_df['day of week'].values == 6)
       bridgeday_df.loc[weekend_mask,'Bridgeday'] = 0             
                     
       
   
    #now remove the days that are already a public or bank holiday
    already_holiday_mask =  (holidaytype_df['Public'] == 1)  \
                          | (holidaytype_df['Bank'] == 1)    
                          
    bridgeday_df.loc[already_holiday_mask,'Bridgeday'] = 0                          
    
    return bridgeday_df

           
           
    
def prep_all_data(country_code, startdate, enddate, 
            resultdir='../data/', loginfile='../userdata/logins.txt' ):
    """
    Function that prepares all the data that is needed for the demand forecast

    Args
    ----------
    country_code : string
        two letter iso country code
    startdate : string
        first date  of the time period in the format 'YYYY-MM-DD'
    enddate : string
        last date  of the time period in the format 'YYYY-MM-DD'
        
    Keyword Args
    ----------    
        
    resultdir : string, default '../data/'
    directory where the data is to be stored
    

    Returns
    -------
    None
                
    
    File output
    ----------
    A csv file for each each dataset. The files have the following names:
        
    <country_code>_<startdate>_to_<enddate>_<description of the data>.csv

    """
    
    filenamebase = f'{resultdir}/{country_code}_{startdate}_to_{enddate}'
                       
    
    # first get the power data from the ENTSO-E platform    
    entsoe_obj = EntsoePower( country_code, startdate, enddate, 
                              loginfile=loginfile )    
    entsoe_filename = f'{filenamebase}_ENTSOE_power.pkl'    
    entsoe_obj.pull_process_save(entsoe_filename)
    
    #get the holidays group them by type
    holiday_obj = OpenHolidays( country_code, startdate, enddate)    
    holiday_filename = f'{filenamebase}_holidays_by_type.pkl'    
    holiday_obj.pull_process_save('holidaytype', filename=holiday_filename)
    
    #determine the bridgedays
    ## first for the case the end of year period after christmas
    ## should also be treated as bridgeday
    bridgedays_df=bridgedays(holiday_obj.holiday_hr, xmas2NY_as_bridge=True)
    bridgeday_filename = f'{filenamebase}_bridgedays_incl_yearend.pkl' 
    bridgedays_df.to_pickle(bridgeday_filename) 
    
    bridgedays_df=bridgedays(holiday_obj.holiday_hr, xmas2NY_as_bridge=False)
    bridgeday_filename = f'{filenamebase}_bridgedays_standard.pkl' 
    bridgedays_df.to_pickle(bridgeday_filename)     
    
    #create a different dataset with the holidays grouped by their name
    holiday_filename = f'{filenamebase}_holidays_by_name.pkl'    
    holiday_obj.json2df('holidayname')
    holiday_obj.to_hourly() 
    holiday_obj.holiday_hr.to_pickle(holiday_filename)
        
    #take the datetime index from the hourly entsoe data
    dt_index = entsoe_obj.data_hr.index
    
    #use the datetime index to generate the time related variables
    ##first treat the hour as a category
    hour_var_as_cat = hour_variable(dt_index, as_category=True)
    hour_filename   = f'{filenamebase}_hour_as_category.pkl'     
    hour_var_as_cat.to_pickle(hour_filename)
    
    ## second treat the hour as a continuous variable
    hour_var_as_var = hour_variable(dt_index, as_category=False)
    hour_filename   = f'{filenamebase}_hour_as_variable.pkl' 
    hour_var_as_var.to_pickle(hour_filename)
    
    # determine the typedays
    ## first the version where the typeday equals the weekday
    weekday_type_day = type_day(dt_index, combine_midweekdays=False)
    typeday_filename = f'{filenamebase}_weekday_as_typeday.pkl' 
    weekday_type_day.to_pickle(typeday_filename)
    
    ##now the version where Tuesday, wednesday and thursday are combined
    combiday_type_day = type_day(dt_index, combine_midweekdays=True)
    combiday_filename  = f'{filenamebase}_combined_weekday_as_typeday.pkl' 
    combiday_type_day.to_pickle(combiday_filename)    
     
    #get the daylight savingstimes flags
    for lag in range(7):
        dls_flags = daylightsaving_flags(dt_index, switchdaylag=lag)
        dls_flag_filename  = ( f'{filenamebase}_daylight_savings_time_flags'
                               f'_lag_{lag}.pkl' )
        dls_flags.to_pickle(dls_flag_filename)    



class ERA5_Weather():
    """
    Class to retrieve and process meteorological data from the ERA5-Land 
    re-analysis dataset 
    
    How to use: generate a ERA5_object (see __init__)
    and apply the desired method(s)
    
    
    contains the following methods:
    --------------------------
    __init__ 
        intialisation of the object
    request_process_save:
         a single method to retrieve the data, process it and save it to
         a pickled dataframe. 
    request_ERA5:
        Splits the requested timeperiod in smaller timeperiods so they are
        small enough for the the ECMWF/Copernicus server,
        calls single_request for de individual requests
        ands saves the results in a series of files
    single_request:
        sends a single request to the ECMWF/Copernicus server
        and saves the result to file     
    multi_weatherfile2df:
        reads the weather data from the series of files
        and combines their content to a pandas dataframe 
    single_weatherfile2df:
        reads a single weatherfile converts it to a pandas dataframe    
    adjust_ERA5_time:
        converts the UTC time of the original ERA5 data to local time and
        adjust the timestamps of the weather dataframe so they becom
        consistent with the ENTSOE data.
    
    """
    
    def __init__(self, startdate, enddate, location, 
                 datafilebasename = '../data/ERA5_weather',
                 variables=['10m_u_component_of_wind', 
                            '10m_v_component_of_wind', 
                             '2m_temperature',
                             'surface_solar_radiation_downwards'],
                              loginfile='../userdata/logins.txt'):
        """
        Args:
        ------------
        startdate: string
            first date of the required period in the format 'YYYY-MM-DD'
        endtdate: string
            last tdate of the required period in the format 'YYYY-MM-DD'
        location: string
            two letter country code as used in 
            country_timezone and country_area
           
        
        Keyword Args:
        -------------
        datafilebasename: string, default = '../data/ERA5_weather'
            will be used to generate names for intermediate files and
            and the filename for the final result
        variables: list of string, default:
                                        ['10m_u_component_of_wind', 
                                         '10m_v_component_of_wind', 
                                         '2m_temperature',
                                         'surface_solar_radiation_downwards'],
        loginfile: string, default: '../userdata/logins.txt'
            file with the api access key
        """
        #read the login data
        #get the range
        
        self.startdate = startdate
        self.enddate   = enddate
        self.location  = location
        self.datafilebasename   = datafilebasename
        self.area = country_area[location]
        self.country_timezone = country_timezone[location]
        self.loginfile = loginfile
        self.variables = variables
        
        self.csdapi_key=read_login(self.loginfile,'ecmwf')
        self.csdapi_url='https://cds.climate.copernicus.eu/api/v2'
        
    
    def request_process_save(self, outfilename=None,
                             variables=['10m_u_component_of_wind', 
                                        '10m_v_component_of_wind', 
                                        '2m_temperature',
                                        'surface_solar_radiation_downwards']):
        """
        requests the weatherdata, processes it and saves it.
        
        Keyword Args:
        -------------    
        outfilename, string or None, default None
            name for the pickle file where the resulting dataframe is stored in
            if None it will be based on the basename given when initialising
            the ERa5 object
        variables: list of strings, default: ['10m_u_component_of_wind', 
                                           '10m_v_component_of_wind', 
                                           '2m_temperature',
                                           'surface_solar_radiation_downwards']
            names or the variables to be retrieved, as defined by the csd_api
            (see: https://cds.climate.copernicus.eu/cdsapp#!/dataset/
             reanalysis-era5-land?tab=overview, make the variable name lower
             score and replace spaces with underscores)


        Returns
        -------
        None.
        
        File output:
        ..............
        
        pickle file with the hourly weather data as pandas dataframe
        
        """
        
        if outfilename == None:
            outfilename = f'{self.datafilebasename}.pk'
        
        
        batch_filenames = self.request_ERA5()
        weather_df = self.multi_weatherfile2df(batch_filenames)
        weather_df = self.adjust_ERA5_time(weather_df, self.country_timezone)
        weather_df.to_pickle(outfilename)
        
    
    def request_ERA5(self):
        """
        Main method to request era 5 data
        
        request  the ERa5 weather data 
        from the ECWMF/Copernicus server in batches.
        the ecwmf/copernicus api can handle only 12000 datapoints per request
        so this method split the requests in multiple requests 
        
        To save time the batch requests are send out to the server in parallel
        using the async library.

        Returns
        -------
        batch_filenames: list of strings
            list with the filenames of the zipped netcdf files containing
            the netcdf data

        """
        #starttime=time.time()
        #print('starttime: ',starttime)   
            
        max_per_batch=12000  #maximum number of data points per api request
        
        #ERA5 data is UTC, but the endgoal is local datetime, 
        # so adding a day at start and end will allow for having complete
        # days at start and end in the local time. Furthermore,
        # we need an additional hour at the end to average "snapshot" data
        #into hourly averages
        request_startdate = pd.to_datetime(self.startdate) - pd.to_timedelta('1D')
        request_enddate = pd.to_datetime(self.enddate) + pd.to_timedelta('1D')
        
        startyear = request_startdate.year
        startmonth = request_startdate.month
        startday = request_startdate.day
        
        endyear =request_enddate.year
        endmonth = request_enddate.month
        endday = request_enddate.day        
        
        
        filename_body, filename_ext= splitext(self.datafilebasename)
                        
        nrofvariables=len(self.variables)
        
        pointspermonth=nrofvariables*31*24
                
        batchnr=0
        batch_filenames=[]
        
        if pointspermonth > max_per_batch:
            raise Exception('Too much points per month')
        else:
            #quick and dirty solution: one calendar month per batch
            for year in range(startyear,endyear+1):
                if year==startyear:
                    firstmonth = startmonth
                else: 
                    firstmonth = 1   
                    
                if year == endyear:
                    lastmonth = endmonth
                else:
                    lastmonth = 12
                    
                for month in range(firstmonth, lastmonth+1):
                    if year==startyear & month == startmonth:
                        firstday = startday
                    else: 
                        firstday =1
                    if year==endyear & month == endmonth:
                        lastday = endday
                    else: 
                        lastday =31 # MARS server truncates the month at
                                    # real length automatically
                                    
                                    
                    requestdays=[f'{i:02}' for i in range(firstday,lastday+1)]                
                    
                    batchnr+=1                                    
                                  
                    batch_filename = (f'{filename_body}_batch_' 
                                      f'{batchnr:0>{3}}'
                                      '.zip')
                    
                    batch_filenames=batch_filenames+[batch_filename]
                    
                                    
                    self.single_request( str(year), 
                                         '{:02}'.format(month),requestdays,
                                          self.area, self.variables,
                                          batch_filename )
            #endtime=time.time()

            return batch_filenames            
  
    
    
    def single_request(self, year, months,days, area, variables, outfilename ):
        """
        helper method to request_ERA5, but can also be run as
        stand allone method
        creates a single request and pass in to the ECMWF MARS server
        
        can also be used by the user to create a request "by hand"
        but in that case they should be aware of the maximum of 12000 data
        points per request, and the pecularities of the cdsapi interface
        
        Args:
        -----
        year: int or string
            year for which data is requested
        months: string or list of strings
            number of the month or the months for which data is requested, 
            single digit months must have a leading zero
        days: list of strings
            numbers of the days in the months for which data is requested
            single digit days must have a leading zero
            "overshooting" the day numbers is not a problem,
            the server ignores day numbers that are longer than the 
            length of the months, for example February 30 is not a problem.
        outfilename: string
            path and name of the resulting zipped netcdf file
        
        
        FILE OUPUT:
        zipped netcdf file containing the retrieved data
        
        """
        
        print(f'*** requesting {year} {months} ****')
        cds_obj = cdsapi.Client(key=self.csdapi_key,url=self.csdapi_url)

        cds_obj.retrieve(
            'reanalysis-era5-land',
            {
                'variable': variables,
                'year':str( year),
                'month': months,
                'time': [f'{i:02}:00' for i in range(0,24)],
                'day': days,
                'area': area,
                'format': 'netcdf.zip',
            },
            outfilename)



    def multi_weatherfile2df(self,batch_filenames) :
        """
        reads a series of zipped netcdf ERA5 files that contains spatially
        gridded timeseries of weather data.
        Their content is processed (see the method single_weatherfile2df)
        and combined to a single pandas dataframe
        

        Args
        ----------
        batch_filenames : list of strings
            List of the filenames that needs to be processed

        Returns
        -------
        df_out

        """
        
        nrof_files =len(batch_filenames)
        df_list=nrof_files *[None]   
                
        for i,batch_filename in enumerate(batch_filenames):
            df_list[i]= self.single_weatherfile2df(batch_filename)
                        
        df_out=pd.concat (df_list)        
        
        return df_out
                

    def single_weatherfile2df(self,batch_filename):
        """
        
        Loads a single zipped netcdf file containing spatially gridded
        ERA5 reanalysis weather data and consequently:
        
        - calculates the total windspeed from the directional components,
          when these directional components are present in the file
        - calculates the hourly solar irradiation  (ssrd) from the cummulative 
          irradiation, when the cummulative ssrd is present in the file
        - calculates the spatial average for all variables
        - converts the data to a pandas dataframe
              
    
        Args:
        ----------
        batch_filename : string
            filename to be loaded
    
        Returns
        -------
        weather_df: pandas dataframe with UTC datetime index
            contains the integrated weather data
    
        """
        #unzip file to working directory
        extractdir='../tempdata/'
        
        shutil.unpack_archive(batch_filename,extractdir)
                
       # print(f'{extractdir}/data.nc')
        weather_ds = xr.open_dataset(f'{extractdir}/data.nc')
        
        #TODO delete data.nc
        
        
        #calculate the magnitude of the windspeed
        # non linear so must be done before averaging
        if 'u10' in weather_ds and 'v10' in weather_ds:
            windspeed = np.sqrt(weather_ds['u10']**2 + weather_ds['v10']**2)
            weather_ds['windspeed'] = windspeed    
            # Remove windspeed components 'u' and 'v'  from the dataset
            weather_ds = weather_ds.drop_vars(['u10', 'v10'])
        
        else:
            raise Warning('cannot calculate magnitude of windspeed'
                          '"u" and/or "v" are missing')
                        
        spat_avg_weather_ds = weather_ds.mean(dim=['latitude', 'longitude'])        

        weather_df = spat_avg_weather_ds.to_dataframe()
        
        #convert cummulative ssrd to hourly values. 
        # is linear process so can be done after averaging
        
        if 'ssrd' in weather_ds:
            ssrd_hrly = np.zeros(weather_df['ssrd'].shape)            
            ssrd_hrly[1:] = weather_df['ssrd'].values[1:] \
                                             - weather_df['ssrd'].values[0:-1]
            #otherwise negative values at 01:00 UTC:                                 
            ssrd_hrly[ssrd_hrly <0 ] = 0.0
            weather_df['ssrd'] = ssrd_hrly
        else:
            raise Warning('ssrd (solar irradiation) not present in dataset')    
        
        return weather_df
        

    def adjust_ERA5_time(self,weather_df, timezone):
        """
        adjusts the timestamps of the weather data to be consistent with 
        the ENTSOE data and converts to local time.
                
        The timestamp of the ENTSOE data indicates the start of the averaging 
        period.
        For the ERA5 irradiation data the timestamp indicates the end of the 
        averaging period. So this data needs to be shifted one hour.
        For the other ERA5 data the timestamp indicated a momenteneous value,
        i.e a snapshot value at a particular timepoint. For these quantities  
        the average over the hour will be approximated by averaging the values of 
        two adjecent time points.
        
    
        Args
        ----------
        weather_df : pandas dataframe 
            contains the ERA5 weather data with the original UTC timestamps
        timezone: string or None
            if None, the timestamps will remain in UTC
            the timezone should be given according to pandas standards
    
        Returns
        -------
        weather_df_out: pandas dataframe
            contains the ERA5 weather data with adjusted timestamps.
    
        """
        
        #create new datetime_index for the data 
        ## from naive to UTC and correct for different timestamp convention
        ## between ERA5 and ENTSOE data
        new_index = weather_df.index.tz_localize('UTC')-pd.to_timedelta('1h')
        
        #convert to local timezone
        if timezone != None:
            new_index = new_index.tz_convert(timezone)
        
        
        weather_df_out = pd.DataFrame(index=new_index, columns=weather_df.columns)
        
        #the irradiance data is a temporal average so can just be copied
        weather_df_out['ssrd'] = weather_df['ssrd'].values
        
       
        # all other variables are snapshots, a temporal average is approximated
        # by averaging the value at start of hour and end of hour.
        snapshot_vars = list(weather_df_out.columns)
        snapshot_vars.remove('ssrd')
        
        
        #get integer indexes of snapshot_vars
        snsh_var_index = [weather_df.columns.get_loc(var) for var in snapshot_vars]
                   
        weather_df_out.iloc[0:-1,snsh_var_index] =  (   
                               weather_df.iloc[0:-1,snsh_var_index].values  
                             + weather_df.iloc[1:,snsh_var_index].values    ) / 2.0
         
          #for last point in the timeseries it is not possible to average.  
        weather_df_out.iloc[-1,snsh_var_index] = weather_df.iloc[-1,snsh_var_index
                                                                           ].values
        
        return weather_df_out




        