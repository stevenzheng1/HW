################################################################
## Packages                                                   ##
################################################################

from IPython.display import clear_output
    
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
#pd.set_option('display.float_format', lambda x: '%.2f' % x)
#pd.set_option('display.max_colwidth', -1)
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import pickle
#from linearmodels import PanelOLS
#from linearmodels.panel import compare
from scipy.stats import mstats as stats
import re
from functools import reduce
import math
import os
from pathlib import Path
import statsmodels.api as sm
from time import sleep
import sys
from datetime import datetime
 
from tqdm import tqdm_notebook as tqdm
#import swifter

## Fuzzy matching
#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process
 
## NPPES API
#from npyi import npi

    
## Edgar
#import edgar; edgar2 = edgar.Edgar()

## Taking integrals
from scipy.integrate import quad
from scipy import stats

import seaborn as sns; sns.set()
from stargazer.stargazer import Stargazer


from sklearn import metrics
#from xgboost.sklearn import XGBClassifier  
#from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

import pycountry

from IPython.core.display import HTML
pd.set_option('display.max_rows', 500)
from sklearn.preprocessing import StandardScaler

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

from dateutil.relativedelta import relativedelta
from statsmodels.regression.rolling import RollingOLS

import datetime as dt
from dateutil.relativedelta import WE, FR


from fredapi import Fred
fred = Fred(api_key='0b5217c99427e4cd8bf33dbde2f18e6d')

from scipy.stats import trim_mean
from scipy.stats.mstats import trimmed_std