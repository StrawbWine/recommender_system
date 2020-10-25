import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle
import timeit
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import TheilSenRegressor

class Content_filtering:
    
    def __init__(
        self, 
        feature_cols,
        user_col='BrukerID',
        item_col = 'FilmID',
        rating_col='Rangering',
        reg_model=LinearRegression()
    ):
        self.model = {} 
        self.reg_model = reg_model
        self.user_col = user_col
        self.item_col = item_col
        self.feature_cols = feature_cols
        self.rating_col = rating_col        
    
    def _content_filtering(self, X, y, reg_model):        
        df = pd.concat([X, y], axis=1)

        def create_model(dfx):
            X = dfx[self.feature_cols]
            y = dfx[self.rating_col]
            reg = copy.copy(reg_model)
            reg.fit(X, y)
            bruker_id = dfx.iat[0, 0]          
            self.model[bruker_id] = reg
            
        df.groupby([self.user_col]).apply(create_model)
        return       
    
    def fit(self, X, y):        
        self._content_filtering(X, y, self.reg_model)
        
    def predict(self, X, return_as_array=True):
        brukerliste = X[self.user_col].drop_duplicates()
        columnlist = X.columns.tolist()        
        columnlist.append(self.rating_col)       
        predictions = pd.DataFrame([], columns=columnlist)
        for bruker in brukerliste:
            X_current_user = X.loc[X[self.user_col] == bruker]
            try:
                y_pred = pd.Series(self.model[bruker].predict(X_current_user[self.feature_cols]))
            except:
                y_pred = pd.Series([3.0]*len(X_current_user.index))
            y_pred.name = self.rating_col            
            X_y_pred = pd.concat( \
                [ \
                    X_current_user[self.user_col].reset_index(drop=True), \
                    X_current_user[self.item_col].reset_index(drop=True), \
                    X_current_user[self.feature_cols].reset_index(drop=True), \
                    y_pred \
                ], \
                axis=1)
            predictions = pd.concat([predictions, X_y_pred], axis=0)            
        predictions[self.rating_col] = predictions[self.rating_col].where(predictions[self.rating_col] <= 5.0, 5.0)        
        predictions[self.rating_col] = predictions[self.rating_col].where(predictions[self.rating_col] >= 1.0, 1.0)
        predictions = predictions.set_index([self.user_col, self.item_col]).reindex(X.set_index([self.user_col, self.item_col]).index)        
        predictions.rename(columns={'value': self.rating_col}, inplace=True)
        if return_as_array:
            return predictions[self.rating_col].values
        else:
            return predictions[self.rating_col]