from Settings import *

def this_clean_all_fff_9900(this_data,this_file,directory_int):
    '''
    After each file is imported, clean it lightly. To speed it up, most cleaning
    is done separately in a notebook FFF/1b. This just subsets data; all 
    creating of columns is done in the notebook to save time.
    
    We do not aggregate to second level as we need to drop duplicates across
    all files. Some dates have multiple files, with duplicates across files.
    So aggregation is done in the notebook FFF/1b.
    '''
    
    # Date
    ## CME/CBOT T&S data is in CT, correct to EST
    this_data['Trade Date'] = pd.to_datetime(this_data['Trade Date'],
                                             format='%Y%m%d') 
    this_data['date'] = this_data['Trade Date'] +\
                        pd.to_timedelta(this_data['Trade Time']) +\
                        pd.to_timedelta(1,unit='h')
    
    ## Actually, keep Trade Date as raw date in CT, otherwise
    ## get maturity <0 issues. It's fine, just go off date for 
    ## ET times and dates
    ## Clean up again, for trades where CT to ET pushes to next day
    #this_data['Trade Date'] = pd.to_datetime(this_data['date'].dt.date)
    
    ## 46 observations with weird delivery date
    ## These all have no volume
    #this_data = this_data[this_data['Contract Delivery Date']!=9900].copy()
    
    ## Drop zero volume as in Francesco's paper
    ## Note: do this before others, as expiry is weird with volume=0
    ## But then dates are 2003 and beyond
    #cond = (this_data['Session Indicator'].str.lower()=='r') |\
    #       ((this_data['Session Indicator'].str.lower()!='r') &\
    #        (this_data['Trade Quantity']>0))
    #this_data = this_data.loc[cond].copy()
    #if len(this_data)==0:
    #    return('no pos')
    
    
    ## Sub
    this_data = this_data[['Trade Date','Contract Delivery Date','Trade Time',
                           'Trade Price','Trade Quantity','date',
                           'Trade Sequence Number','Session Indicator']].copy()
    
    ## Sort
    this_data.sort_values(['Trade Date','Contract Delivery Date','Trade Time'],
                          inplace=True)
    
    ## Out
    this_data.to_parquet(directory_int+this_file+'.parquet')
    return('success')
    
    
def this_load_all_fff_9900(this_file,directory_raw,directory_int):
    '''
    Same as this_load_fff() but for all data.
    '''
    
    try:
    
        ## Import it
        #this_data = this_import(this_file)
        this_data = this_import(this_file=this_file,
                                directory_raw=directory_raw)

        ## Clean it and out
        ## Note, this is all mutable
        this_result = this_clean_all_fff_9900(this_data=this_data,
                                         this_file=this_file,
                                         directory_int=directory_int)

        return(this_result)

    except Exception as e:
        return('fail')