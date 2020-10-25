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

class Collab_filtering:    
    
    def __init__(self, n_buddies=50, use_weights=False, user_col='BrukerID', item_col='FilmID', rating_col='Rangering'):        
        self.model = None
        self.n_buddies = n_buddies
        self.use_weights = use_weights
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col        

    # Private methods    
        
    def _to_relative(self, df):
        df_user_mean_ratings = df.groupby([self.user_col])[self.rating_col].mean()
        df_user_mean_ratings.name = 'User_mean_rating'
        df_out = pd.merge(
            df,
            df_user_mean_ratings,
            how='inner',
            on=self.user_col)
        df_out[self.rating_col] = df_out[self.rating_col] - df_out['User_mean_rating']
        return df_out
    
    def _to_wide(self, df):        
        df_rel = df[[self.user_col, self.item_col, self.rating_col]]
        df_rel_wide = df_rel.pivot(index=self.user_col, columns=self.item_col, values=self.rating_col)
        return df_rel_wide
    
    def _weighted_avg(self, values, weights):
        w_avg = (values * weights).sum() / weights.sum()
        return w_avg
    
    def _get_buddies(self, df, n_buddies):
        df = df.fillna(0.0)
        df_corrs = df.T.corr()
        buddies = {}
        for bruker in df.index:    
            buddies[bruker] = df_corrs[bruker].nlargest(n_buddies)[1:]
        return buddies

    def _collab_filtering(self, df, use_weights):
        means = self._to_wide(df).T.mean()
        df_wide = self._to_wide(self._to_relative(df))        
        df_work = df_wide.copy()
        all_buddies = self._get_buddies(df_wide, self.n_buddies)
        if use_weights:            
            def get_weighted_avg_buddy_ratings(row):
                my_buddies = all_buddies[row.name].reset_index()[self.user_col]
                weights = all_buddies[row.name]/all_buddies[row.name].mean()
                buddy_weighted_avg_ratings = self._weighted_avg(df_wide.loc[list(my_buddies), :], weights)
                return buddy_weighted_avg_ratings
            df_work = df_wide.apply(get_weighted_avg_buddy_ratings, axis=1)
            df_out = df_wide.where(df_wide.notna(), df_work)
            df_out = df_out.fillna(0.0)                
        else:
            def get_avg_buddy_ratings(row):
                my_buddies = all_buddies[row.name].reset_index()[self.user_col]
                buddy_avg_ratings = df_wide.loc[list(my_buddies), :].mean()
                return buddy_avg_ratings
            df_work = df_wide.apply(get_avg_buddy_ratings, axis=1)
            df_out = df_wide.where(df_wide.notna(), df_work)
            df_out = df_out.fillna(0.0)
        df_out = df_out.add(means, axis=0)    
        return df_out
    
    # Public methods
    
    def fit(self, X, y):
        df = pd.concat([X, y], axis=1)        
        self.model = self._collab_filtering(df, self.use_weights)
        
    def predict(self, df, return_as_array=True):
        melted = pd.melt(self.model.reset_index(), id_vars=[self.user_col], value_vars=self.model.columns).dropna()
        melted = melted.reset_index(drop=True)
        predictions = melted.set_index([self.user_col, self.item_col]).reindex(df.set_index([self.user_col, self.item_col]).index)        
        predictions.rename(columns={'value': self.rating_col}, inplace=True)
        predictions = predictions['Rangering'].fillna(predictions['Rangering'].dropna().mean())
        if return_as_array:
            return predictions.values
        return predictions