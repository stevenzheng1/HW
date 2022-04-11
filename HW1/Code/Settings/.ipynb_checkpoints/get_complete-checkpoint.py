from Settings import *

def get_index(date_start,date_end):

    ## Set up hours
    ## Trading hours are Sunday 6pm - Friday 5pm
    ## Closed Monday to Thursday 5-6pm
    new_index = pd.DataFrame(
                    {'date':pd.date_range(date_start,
                                          date_end,
                                  freq='1min')})
    new_index['day'] = new_index['date'].dt.dayofweek+1
    new_index['hour'] = new_index['date'].dt.hour
    new_index['flag_drop'] = 0
    new_index.loc[(new_index['day']==6) | 
                  ((new_index['day']==7) & (new_index['hour']<18)) |
                  ((new_index['day']==5) & (new_index['hour']<17)) |
                  ((new_index['day'].isin([1,2,3,4])) & (new_index['hour']==17)),
                  'flag_drop']=1
    new_index = new_index[new_index['flag_drop']==0].copy()
    new_index.drop(columns=['day','hour','flag_drop'],
                   inplace=True)
    new_index.set_index('date',inplace=True)
    
    return(new_index.index)


def make_complete_ffill(df):
    
    new_index = get_index(date_start='1995-01-03 00:00:00',
                          date_end='2020-06-02 23:59:59')

    df = df.reindex(new_index,
                    method='ffill')
    return(df)

def make_complete_notffill(df):
    
    new_index = get_index(date_start='1995-01-03 00:00:00',
                          date_end='2020-06-02 23:59:59')

    df = df.reindex(new_index)
    return(df)


def this_complete(this_mat,data_m_avg,subfolder_intdata,method,data_m_last=''):
    
    try:
        
        this_data_m = data_m_avg[data_m_avg['maturity']==this_mat].copy()
        
        if method=='avg_ffill':
            ## Complete and ffill the last average
            this_data_c = this_data_m.sort_values('date')\
                                     .set_index('date')\
                                     .groupby('maturity')\
                                     .apply(make_complete_ffill)\
                                     .drop(columns='maturity')\
                                     .reset_index()\
                                     .rename(columns={'level_1':'date'})
        if method=='avg_last':
            ## Complete and rolling joing last last
            this_data_c = this_data_m.sort_values('date')\
                                     .set_index('date')\
                                     .groupby('maturity')\
                                     .apply(make_complete_notffill)\
                                     .drop(columns='maturity')\
                                     .reset_index()\
                                     .rename(columns={'level_1':'date'})
            this_data_c.sort_values('date',
                                    inplace=True)
            
            ## Rolling join left
            data_m_last.sort_values('date',
                                   inplace=True)
            this_data_c = pd.merge_asof(
                              left=this_data_c,
                              right=data_m_last,
                              by='maturity',
                              left_on='date',
                              right_on='date',
                              allow_exact_matches=True,
                              direction='backward')
            
            this_data_c['f'] = this_data_c['f_avg'].copy()
            cond = (this_data_c['f_avg'].isna())
            this_data_c.loc[cond,'f'] = this_data_c.loc[cond]['f_last'].copy()
            
        this_data_c.to_parquet(subfolder_intdata+'maturity'+str(this_mat)+'.parquet')
        return('success')
    
    except Exception as e:
        return('fail')