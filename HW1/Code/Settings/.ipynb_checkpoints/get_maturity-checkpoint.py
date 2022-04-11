from Settings import *

def get_maturity(data_in,product,date_col='Trade Date'):
    '''
    Calculate maturity. It's an int where 0 is the nearest expiry,
    1 is the next etc. For FFF, this coincides with number of months
    until expiry, where 0 is current month-end.
    
    For ED, it's only for quarterly contracts. Make sure serial ones
    are dropped before calculating maturity.
    
    For SP500F, there's contracts for 9 consecutive quarters and 3
    additional December contract months. So keep all contracts and
    run this code to get maturity, then drop all those with
    maturity>8 (since 0 is the first contract, so 8 is the 9th).
    
    date_col='Trade Date' most of the time, except for rolling join of 
    FFF or ED on FOMC, when it's 'fomc'.
    '''
    
    ## Make it local
    data = data_in.copy()
    
    data['maturity'] = 12*(data['expiry'].dt.year-\
                           data[date_col].dt.year) +\
                       (data['expiry'].dt.month-\
                        data[date_col].dt.month)

    if product=='ed' or product=='sp500f':
        data.rename(columns={'maturity':'maturity_month'},
                      inplace=True)
        
        tmp = data[[date_col]].copy().drop_duplicates()
        tmp[date_col+' ym'] = tmp[date_col] + pd.offsets.MonthEnd(0)
        
        if product=='ed':
            tmp['third_monday'] = tmp[date_col+' ym'].apply(get_ed_date)
        if product=='sp500f':
            ## Actually third Friday, but makes it easy below
            tmp['third_monday'] = tmp[date_col+' ym'].apply(get_sp500f_date)
        
        tmp['flag_afterexpiry'] = 0
        tmp.loc[(tmp[date_col].dt.month.isin([3,6,9,12])) &
                (tmp[date_col]>tmp['third_monday']),
                'flag_afterexpiry'] = 1
        data = pd.merge(left=data,
                        right=tmp[[date_col,'third_monday','flag_afterexpiry']],
                        on=[date_col],
                        how='left')
        data['maturity'] = (data['maturity_month']/3).astype(int)
        cond = (data['flag_afterexpiry']==1)
        data.loc[cond,'maturity'] = data.loc[cond,'maturity'].copy()-1
        data.drop(columns=['maturity_month','third_monday','flag_afterexpiry'],
                    inplace=True)
        
    ## Out
    return(data)