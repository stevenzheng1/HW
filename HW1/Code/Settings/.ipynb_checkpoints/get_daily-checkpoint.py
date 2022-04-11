from Settings import *

def get_daily(data_min,this_mat,save_dir):
    
    try:
        this_data_min = data_min[data_min['maturity']==this_mat].copy()
        this_data_daily = this_data_min.set_index('date')\
                                       .groupby('maturity')\
                                       .resample('D')\
                                       .last()\
                                       .drop(columns=['maturity'])\
                                       .reset_index()
        this_data_daily.to_parquet(save_dir+
                                   'this_data_daily_'+
                                   str(this_mat)+
                                   '.parquet')
        return('success')
    except Exception as e:
        return('fail')
