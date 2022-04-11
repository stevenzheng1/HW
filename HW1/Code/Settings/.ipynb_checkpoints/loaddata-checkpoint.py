from Settings import *

def this_import(this_file,directory_raw):   
    ''' 
    This just imports each raw file. Each raw file is a date, with transactions
    at the second level. We'll import it and sub on the columns and return it.
    '''
        
    ## Layout from 
    ## https://www.cmegroup.com/confluence/display/EPICSANDBOX/Time+and+Sales
    names = ['Trade Date',
             'Trade Time',
             'Trade Sequence Number',
             'Session Indicator',
             'Ticker Symbol',
             'Future/Option/Index Indicator',
             'Contract Delivery Date',
             'Trade Quantity',
             'Strike Price',
             'Trade Price',
             'Ask Bid Type',
             'Indicative Quote Type',
             'Market Quote',
             'Close Open Type',
             'Valid Open Exception',
             'Post Close',
             'Cancel Code Type',
             'Insert Code Type',
             'Fast Late Indicator',
             'Cabinet Indicator',
             'Book Indicator',
             'Entry Date',
             'Exchange Code']
    
    ## In
    this_data = pd.read_csv(directory_raw+this_file,
                            header=None,
                            names=names)
    
    ## Sub
    cols = ['Trade Date','Trade Time','Trade Sequence Number',
            'Contract Delivery Date','Trade Price',
            'Trade Quantity','Session Indicator']
    this_data = this_data[cols].copy()
    
    ## Out
    return(this_data)


def get_ed_date(this_date):
    '''
    Get the Monday before the third Wednesday in the month
    of this_date, where this_date is in month-end just
    out of covenience.
    '''
    
    this_date_ed = this_date +\
                   relativedelta(day=1,weekday=WE(3)) -\
                   dt.timedelta(days=2)
    return(this_date_ed)


def this_clean(this_data,this_file,directory_int,product):
    '''
    This takes the imported date file that's at the date-expiry-transaction
    (second) level and turns it into date-expiry-transaction (minute) level
    where it's aggregated by last. Technically we don't need to do this, as 
    we'll be doing rolling joins. But it does reduce the size of the data
    which is easier to look at.
    
    It also cleans up the expiry, subs on expiry
    
    Note: No need to complete and forward fill, as we'll use rolling joins
    on the FOMC dates.
    
    Note: taking advantage of mutability to save memory.
    '''
    
    ## Date
    ## CME/CBOT T&S data is in CT, correct to EST
    this_data['Trade Date'] = pd.to_datetime(this_data['Trade Date'],
                                             format='%Y%m%d') 
    this_data['date'] = this_data['Trade Date'] +\
                        pd.to_timedelta(this_data['Trade Time']) +\
                        pd.to_timedelta(1,unit='h')
    
    ## Observations with weird delivery date
    ## These all have no volume
    this_data = this_data.loc[this_data['Contract Delivery Date']!=9900].copy()
                        
    ## Drop zero volume as in Francesco's paper
    ## Note: do this before others, as expiry is weird with volume=0
    ## But then dates are 2003 and beyond
    cond = (this_data['Session Indicator'].str.lower()=='r') |\
           ((this_data['Session Indicator'].str.lower()!='r') &\
            (this_data['Trade Quantity']>0))
    this_data = this_data.loc[cond].copy()
    if len(this_data)==0:
        return('no pos')
    
    ## Sub
    cols = ['Trade Date','Contract Delivery Date','Trade Time',
            'Trade Price','Trade Quantity','Session Indicator','date',
            'Trade Sequence Number']
    this_data = this_data[cols].copy()

    ## Out
    this_data.to_parquet(directory_int+this_file+'.parquet')
    
    return('success')
    
    
def this_load(this_file,directory_raw,directory_int,product):
    '''
    This is the wrapper function that imports each file and aggregates
    it to the date-expiry-second level.
    
    Set product='ed' for eurodollars. Set it as anything else for fed
    fund futures.
    
    Set volume='pos' for keeping only volume>0 transactions. Set it
    as anything else to keep all volumes (i.e., 0 volume ones too).
    This updated version accounts for the P session indicator.
    
    Note: no data is returned, since it's imported, cleaned (without 
    copying) and outputed. We'll import the intermediate files
    and concat. It's not actually necessary to do it this way since
    the files aren't that big.
    '''
    
    try:
    
        ## Import it
        this_data = this_import(this_file=this_file,
                                directory_raw=directory_raw)

        ## Clean it and out
        ## Note, this is all mutable
        this_result = this_clean(this_data=this_data,
                                 this_file=this_file,
                                 directory_int=directory_int,
                                 product=product)

        return(this_result)

    except Exception as e:
        return('fail')
    