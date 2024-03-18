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
__version__    = "0.01"
__maintainer__ = "Jethro Betcke"


# TODO add class to pull weather data (openweather/ECMWF/ERA5)
# TODO holidays: select only national holidays, or make an option for this
# TODO holidays: workaround to deal with the maximum of 3 years of data 
#                that can be pulled from openholidaysapi.org
# TODO bridgedays: create category for days between holiday and weekend
# TODO make structure and names of the different classes more similar

import os
import requests
import pandas as pd
import numpy as np
import time
import pickle
from entsoe import EntsoePandasClient



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

# HELPER FUNCTIONS AND UTILITIES


def one_hot_df(categorical_df, column_prefix=''):
    """
    Applies one hot encoding to a pandas dataframe.
    The motivation to notus keras'to_categorical() is that we want to 
    preserve the category names. Which makes it easier to reuse the model
    
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
        self.startdate = startdate
        self.enddate   = enddate
         
        self.startdate_str = startdate
        self.enddate_str   = enddate
        
        
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
        
        requestline = (f'https://openholidaysapi.org/{holidaytype}'
                       'Holidays?'
                       f'countryIsoCode={self.country_code}&' 
                       'languageIsoCode=EN&'
                       f'validFrom={self.startdate_str}&'
                       f'validTo={self.enddate_str}')
        
        
        response = requests.get(requestline)
        
        if response.status_code != 200:
            raise Exception('Data request to openholidaysapi.org '
                            f'failed with code {response.status_code}')
        
        
        
        return response.json()
    
    
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
                
           
                
        
        #first bring  the dates from tz naive to local timezone
        
        self.holiday_day = holiday_df.tz_localize(self.time_zone)
        
    
    def to_hourly(self):
        """
        resample the daily holiday flags to an hourly resolution.
        
        Returns
        -------
        None.

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
        

           
        


class ERA5Weather:
    """
    class to obtain the ECMWF ERA5 reanalysis weather data
    """    

    def init (startdatetime,enddatetime,lat,lon, 
                   loginfile='../userdata/logins.txt'):

        pass
    
    
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
    None.
                
    
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
    typeday_filename  = f'{filenamebase}_combined_weekday_as_typeday.pkl' 
    weekday_type_day.to_pickle(typeday_filename)    
     
    #get the daylight savingstimes flags
    for lag in range(7):
        dls_flags = daylightsaving_flags(dt_index, switchdaylag=lag)
        dls_flag_filename  = ( f'{filenamebase}_daylight_savings_time_flags'
                               f'_lag_{lag}.pkl' )
        dls_flags.to_pickle(dls_flag_filename)    




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




        