from Settings import *


def rolling_join(fomc_in,data):
    '''
    Rolling join FFF or ED onto FOMC. First cross by expiry, then left and right.
    
    Note: this is by expiry, so later on, outside of this function, get maturity
    again. Not ideal but whatever.
    
    Note: edit data locally since it's so big. FOMC we'll make a copy.
    '''
    
    ## Local copies of mutable parameters
    fomc = fomc_in.copy()
    #data = data_in.copy()
    
    
    ## Cross
    fomc['key']=0
    maturity = data[['expiry']].copy().drop_duplicates()
    maturity['key']=0
    fomc_mat = pd.merge(left=fomc,
                        right=maturity,
                        on='key',
                        how='outer')
    fomc_mat.drop(columns=['key'],
                  inplace=True)
    #fomc.drop(columns=['key'],
    #          inplace=True)
    
    ## Left window
    data.sort_values(['date'],
                     inplace=True)
    
    data_join = pd.merge_asof(
                left=fomc_mat,
                right=data.rename(columns={'price':'price_left',
                                           'date':'date_left'}),
                by='expiry',
                left_on='fomc_left',
                right_on='date_left',
                allow_exact_matches=True,
                direction='backward')

    ## Right window
    data_join = pd.merge_asof(
                left=data_join,
                right=data.rename(columns={'price':'price_right',
                                           'date':'date_right'}),
                by='expiry',
                left_on='fomc_right',
                right_on='date_right',
                allow_exact_matches=True,
                direction='forward',
                suffixes=['_left','_right'])
    
    data_join = data_join[(data_join['price_left'].notna()) &
                          (data_join['price_right'].notna())].copy() 
    
    ## Note: because observatoins are at the date-expiry level,
    ## if the left date month != right date month, then 
    ## matuirty left != maturity right. Big deal since outer window
    ## will drop these any ways
    data_join.drop(columns=['maturity_left','maturity_right'],
                   inplace=True)
    
    
    ## Out
    return(data_join)


def get_surprises(data_in,fomc_meetings_in):
    '''
    This calculates the surprises for FFF. Notation largely follows
    Gurkaynak 2005. Also see Kuttner 2001 and Nakamura and Steinsson
    for more details.
    
    Note: for now, only do scheduled meetings.
    '''
    
    ## Local copies of mutable parameters
    data = data_in.copy()
    fomc_meetings = fomc_meetings_in.copy()
        
    ## Get next month's change in implied rate, for surprises
    tmp_nextmonth = data[['fomc','maturity','delta_f']].copy()
    tmp_nextmonth.rename(columns={'maturity':'maturity_old'},
                         inplace=True)
    tmp_nextmonth['maturity'] = tmp_nextmonth['maturity_old'].copy()-1
    tmp_nextmonth.rename(columns={'delta_f':'delta_f_nextmonth'},
                         inplace=True)
    tmp_nextmonth.drop(columns=['maturity_old'],
                       inplace=True)
    data = pd.merge(left=data,
                    right=tmp_nextmonth,
                    how='left',
                    on=['fomc','maturity'])
    
    ## Have to make a copy, due to not all maturities having a 
    ## fomc meeting. Then merge back on. So tmp has all fomc
    ## meetings in fomc, but only expiries in months where
    ## there is a future FOMC meeting
    tmp = data.copy()

    ## Merge on future FOMC in month of expiry
    ## FFF already EOM but whatever
    tmp['expiry_ym'] = tmp['expiry'] + pd.offsets.MonthEnd(0)
    tmp = pd.merge(left=tmp,
                   right=fomc_meetings,
                   left_on='expiry_ym',
                   right_on='fomc_future_ym',
                   how='inner')
    ## maturity_e is re-indexed maturity, excluding
    ## contracts without a fomc in that month
    ## note the inner join above
    ## tmp has scheduled and unscheduled meetings, which is 
    ## fine since the FOMC matched to each expiry is scheduled only
    tmp['maturity_e'] = tmp.groupby(['fomc'])\
                           ['expiry']\
                           .rank()\
                           -1.0
    tmp['maturity_e'] = tmp['maturity_e'].astype(int)

    ## Number of days in fomc meeting month
    tmp['m'] = tmp['fomc_future_ym'].dt.day

    ## Day of month in fomc meeting month
    tmp['d'] = tmp['fomc_future'].dt.day
    
    ## Surprises
    ## Will create tmp e_horizon, then fill e
    tmp['e'] = np.nan

    ## Surprise for current month's FOMC meeting
    tmp['e_0'] = tmp['delta_f'] * tmp['m']/(tmp['m']-tmp['d'])
    cond = (tmp['maturity_e']==0)
    tmp.loc[cond,'e'] = tmp.loc[cond,'e_0'].copy()
    cond = (tmp['d']>=(tmp['m']-7)) &\
           (tmp['maturity_e']==0) 
    tmp.loc[cond,'e'] = tmp.loc[cond,'delta_f_nextmonth'].copy()    
    
    ## Surprise for next 7 FOMC meetings
    for j in range(1,8):
        tmp.sort_values(['fomc','expiry'],
                        inplace=True)
        tmp['e_shift'] = tmp.groupby(['fomc'])\
                            ['e']\
                            .shift()

        tmp['e_'+str(j)] = (tmp['delta_f'] - tmp['d']/tmp['m']*tmp['e_shift']) *\
                           tmp['m']/(tmp['m']-tmp['d'])
        cond = (tmp['maturity_e']==j)
        tmp.loc[cond,'e'] = tmp.loc[cond,'e_'+str(j)].copy()

        cond = (tmp['d']>=(tmp['m']-7)) &\
               (tmp['maturity_e']==j) 
        tmp.loc[cond,'e'] = tmp.loc[cond,'delta_f_nextmonth'].copy()    
    
    ## Only do first 8 meetings
    tmp.loc[tmp['maturity_e']>7,'e'] = np.nan

    ## Merge back on
    data = pd.merge(left=data,
                    right=tmp[['fomc','expiry','maturity_e','e']],
                    on=['fomc','expiry'],
                    how='left')
    
    ## Sub
    data.drop(columns=['delta_f_nextmonth'],
              inplace=True)
    
    ## Need to drop a bunch of stuff
    return(data)


def get_window_inner(fomc,window_inner):
    '''
    Get inner window, in place, on fomc data.
    '''
    
    if window_inner=='30':
        window_inner_left=10
        window_inner_right=20
    if window_inner=='60':
        window_inner_left=15
        window_inner_right=45
    fomc['fomc_left'] = fomc['fomc'].copy() -\
                        dt.timedelta(minutes=window_inner_left)
    fomc['fomc_right'] = fomc['fomc'].copy() +\
                         dt.timedelta(minutes=window_inner_right)
    

def get_window_outer(data_join,window_outer,product):
    '''
    Get outer window, in place, on joined data.
    '''
    
    
    if window_outer=='ej':
        data_join['outer_left'] = pd.to_datetime(data_join['fomc'].dt.date) +\
                               pd.to_timedelta('00:00:01')
        data_join['outer_right'] = pd.to_datetime(data_join['fomc'].dt.date) +\
                                   pd.to_timedelta('36:00:00')
        cond = (data_join['date_left']<data_join['outer_left']) |\
               (data_join['date_right']>data_join['outer_right'])
    if window_outer=='1h':
        data_join['outer_left'] = np.nan
        data_join['outer_right'] = data_join['fomc_right'] +\
                                   pd.to_timedelta('01:00:00')
        cond = (data_join['date_right']>data_join['outer_right'])
        
    if window_outer=='2h':
        data_join['outer_left'] = np.nan
        data_join['outer_right'] = data_join['fomc_right'] +\
                                   pd.to_timedelta('02:00:00')
        cond = (data_join['date_right']>data_join['outer_right'])
        
    data_join['flag_outer'] = 0
    data_join.loc[cond,'flag_outer'] = 1

    data_join.loc[data_join['flag_outer']>0,'delta_f'] = 0.0
    
    if product=='fff':
        data_join.loc[data_join['flag_outer']>0,'e'] = 0.0
    


def join_fomc(fomc_in,fomc_meetings_in,data_in,window_inner,window_outer,product):
    '''
    This rolling joins either the FFF or ED data onto the FOMC data.
    
    The FOMC data first sets the inner window. Then rolling join. Then cut
    based on outer window.
    
    Then calculate change in implied rate for ED and FFF. For FFF, further
    calculate surprise.
    
    Product is either 'fff' or 'ed'.
    
    window_inner is either '30' or '60'.
    
    window_outer is either 'ej', '1h' or '2h'.
    
    '''
    
    ## Local copies of mutable parameters
    fomc = fomc_in.copy()
    fomc_meetings = fomc_meetings_in.copy()
    data = data_in.copy()
    
    ## Set up inner window
    get_window_inner(fomc=fomc,
                     window_inner=window_inner)
    
    ## Rolling join on expiry, then add in maturity again
    data_join = rolling_join(fomc_in=fomc,
                             data=data)
    
    ## Calculate maturity again
    data_join = get_maturity(data_in=data_join,
                             product=product,
                             date_col='fomc')
    
    ## Change in implied rate (negative price change)
    data_join['delta_f'] = data_join['f_right'].copy() -\
                           data_join['f_left'].copy()
    
    ## Get surprises if FFF
    if product=='fff':
        data_join = get_surprises(data_in=data_join,
                                  fomc_meetings_in=fomc_meetings)
    
    ## Outer window
    get_window_outer(data_join=data_join,
                     window_outer=window_outer,
                     product=product)
    
    ## Out
    return(data_join)