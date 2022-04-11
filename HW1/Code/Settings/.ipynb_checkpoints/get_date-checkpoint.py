from Settings import *

#def get_date(this_date,product):
#    '''
#    Get dates by product.
#
#    For FFF: just keep month end.
#
#    For ED: get the Monday before the third Wednesday 
#    in the month of this_date, where this_date is in 
#    month-end just out of covenience.
#
#    For SP500F: Get the third Friday in the month of
#    this_date, where this-date is in month-end just 
#    out of convenience.
#    '''
#
#    if product=='fff':
#        this_date_clean = this_date
#    
#    if product=='ed':
#        this_date_clean = this_date +\
#                          relativedelta(day=1,weekday=WE(3)) -\
#                          dt.timedelta(days=2)
#
#    if product=='sp500f':
#        this_date_clean = this_date +\
#                          relativedelta(day=1,weekday=FR(3))
#
#    return(this_date_clean)


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


def get_sp500f_date(this_date):
    '''
    Get the third Friday in the month of this_date, where
    this-date is in month-end just out of convenience.

    "Trading terminates at 9:30 a.m. ET on the 3rd Friday
    of the contract month."
    '''
    
    this_date_sp500f = this_date +\
                       relativedelta(day=1,weekday=FR(3))
    return(this_date_sp500f)