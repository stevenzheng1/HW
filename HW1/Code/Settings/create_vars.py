from Settings import *

def create_vars(data_s_in,product):
    '''
    Take in second-level data and clean expiry, maturity, price,
    and implied rate. Drop cleaned prices < 90.
    
    Note: data_s comes in as second level, where we keep the trade 
    with the lowest trade sequence number for trades in the same
    second.
    
    We do not aggregate to minute level.
    '''
    
    ## Local copies of mutable parameters
    data_s = data_s_in.copy()
    
    ################
    ## Get expiry ##
    ################
    
    ## Note: not month end for ED. Also get rid of non-quarterly for ED
    data_s['expiry_raw'] = data_s['Contract Delivery Date']\
                           .astype(str)\
                           .str.rjust(4,'0')

    data_s['yy'] = data_s['expiry_raw'].copy().str[:2]
    data_s['mm'] = data_s['expiry_raw'].copy().str[2:]

    data_s['yyyy'] = np.nan
    data_s.loc[data_s['yy'].astype(int)>=90,'yyyy'] = \
        '19'+data_s.loc[data_s['yy'].astype(int)>=90,'yy'].copy()
    data_s.loc[data_s['yy'].astype(int)<90,'yyyy'] = \
        '20'+data_s.loc[data_s['yy'].astype(int)<90,'yy'].copy()

    data_s['expiry'] = pd.to_datetime(data_s['yyyy']+'-'+\
                                      data_s['mm']+'-01') +\
                          pd.offsets.MonthEnd(0)
    
    if product=='ed':
        data_s['expiry'] = data_s['expiry'].apply(get_ed_date)
        
        ## Keep only quarterly contracts, drop serial ones
        ## Note: make sure this is before maturity for ED
        data_s = data_s.loc[data_s['expiry'].dt.month.isin([3,6,9,12])].copy()

    if product=='sp500f':
        data_s['expiry'] = data_s['expiry'].apply(get_sp500f_date)

        ## This might be wrong
        ## Note: make sure this is before maturity for SP500F
        #data_s = data_s.loc[data_s['expiry'].dt.month.isin([3,6,9,12])].copy()
        
    data_s.drop(columns=['Contract Delivery Date',
                         'expiry_raw','yyyy','yy','mm'],
                inplace=True)
    
    ##################
    ## Get maturity ##
    ##################

    ## Just because we're comparing EFFR on expiry,
    ## which could be pretty far out for some
    data_s = get_maturity(data_in=data_s,
                          product=product)
    
    


    ############
    ## Prices ##
    ############

    if product=='sp500f':
        ## price is just Trade Price, for compatability
        data_s['price'] = data_s['Trade Price'].copy()

        ## Do this around FOMÇ window!
        ## return... 5% is 0.05
        #data_s['ret'] = data_s['price'].copy()/data_s['price'].copy().shift() - 1.0

        ## log returns
        #data_s['log_ret'] = np.log(data_s['price'].copy()) -\
        #                    np.log(data_s['price'].copy().shift())


    if product=='fff' or product=='ed':
        ## Now exclude if price >101 or <90, but earlier before agg
        ## Price, scaled
        ## Note: some reason they're like >100 sometimes. So scale it down
        data_s['price'] = data_s['Trade Price'].copy() /\
                        10**(data_s['Trade Price'].astype(int)\
                                                    .astype(str)\
                                                    .str.len()\
                            -2)
        cond = (data_s['Trade Price']>=100.0) &\
            (data_s['Trade Price']<=101.0)
        data_s.loc[cond,'price'] = data_s.loc[cond,'Trade Price'].copy()
        
        ## Exclude if scaled price < 90
        ## i.e., if raw price is like 5000
        data_s = data_s[data_s['price']>90].copy()
        
        ## FFF implied rate
        data_s['f'] = 100.0 - data_s['price'].copy()
    
    ## Out
    return(data_s)