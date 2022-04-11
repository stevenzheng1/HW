from Settings import *

def mankiw_shapiro_quick_1(seed,N,e_sim_cov,phi_data):
    '''
    Add some details. Whatever.
    '''
    
    ## Set seed
    rng = np.random.default_rng(seed)

    ## Draw e_r and e_pd from size-N bivarioate normal
    ## variances=1, covariance= corr(e_r,e_pd)
    ## ordered as e_r, e_pd
    e_sim = rng.multivariate_normal(mean=[0,0],
                                    cov=e_sim_cov,
                                    size=N)
    e_r_sim = e_sim[:,0]
    e_pd_sim = e_sim[:,1]

    ## Set Y=v_t
    ## Set r = e_r_sim
    r_sim = e_r_sim

    ## Generate X+t from eq 3 using innovations e
    ## Generate pd_t from pd AR(1) using innovations e_pd
    ## Initial value pd_0 is random from univariate normal, mean 0 and variance 1/(1-phi^2)
    ## Note: initial value gets used so we keep number of observations = N in the regression
    ## of pd(t) on pd(t-1)
    pd_sim_0 = rng.normal(loc=0,
                          scale=np.sqrt(1.0/(1.0-phi_data**2))) # SD
    pd_sim_panda = pd.DataFrame({'e_pd':e_pd_sim})
    pd_sim_panda['pd'] = np.nan
    pd_sim_panda.loc[0,'pd'] = pd_sim_0
    for t in range(1,len(pd_sim_panda)+2):
        if t==0:
            pd_sim_tm1 = pd_sim_0
        if t>0:
            pd_sim_tm1 = pd_sim_panda.loc[t-1,'pd'].copy()
        if t==len(pd_sim_panda)+1:
            pd_sim_panda.loc[t,'pd'] = np.nan
            break
        pd_sim_panda.loc[t,'pd'] = phi_data * pd_sim_tm1 + pd_sim_panda.loc[t-1,'e_pd'].copy()
    pd_sim_panda['pd_tm1'] = pd_sim_panda['pd'].copy().shift()
    pd_sim_panda.dropna(subset=['pd','pd_tm1'],
                        inplace=True)
    pd_sim = pd_sim_panda['pd'].copy().values
    pd_sim_tm1 = pd_sim_panda['pd_tm1'].copy().values

    ## Estimate equation 2, grab t-stat
    ## Estimate return regression using generated data
    reg_r_sim = sm.OLS(endog=r_sim,
                       exog=pd_sim,)\
                .fit()

    ## Estimate pd regression using generated data
    reg_pd_sim = sm.OLS(endog=pd_sim,
                        exog=pd_sim_tm1,)\
                .fit()
                
    ## Grab regression coefficients
    b_r_sim = reg_r_sim.params[0]
    phi_sim = reg_pd_sim.params[0]

    
    ## Out
    return(b_r_sim,phi_sim)

def mankiw_shapiro_quick(M,N,e_sim_cov,phi_data):
    '''
    M = number of runs
    N = size of each sun (i.e., size of data, in this case liek 75 or something)
    '''

    ## All runs of M
    results = pd.DataFrame({'seed':range(1,M+1)})
    results[['b_r_sim','phi_sim']] = results.apply(lambda x: 
                                      mankiw_shapiro_quick_1(seed=x['seed'],
                                                       N=N,
                                                       e_sim_cov=e_sim_cov,
                                                       phi_data=phi_data),
                                      axis=1,
                                      result_type='expand')

    ## Summary stats
    stats = results[['b_r_sim','phi_sim']]\
                   .agg(['mean','std'])

    ## Out
    d_out = {'stats':stats}
    return(d_out)