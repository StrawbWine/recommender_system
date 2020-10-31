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

from ../collab_filtering import Collab_filtering
from ../content_filtering import Content_filtering

class Recommender_system:
    
    def __init__(
        self,        
        feature_cols,
        user_col,
        item_col,
        rating_col,
        n_features_content,
        n_features_collab,
        n_buddies,
        content_f_reg_model,
        cont_collab_mix
    ):        
        self.feature_cols = feature_cols
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col        
        self.n_buddies = n_buddies
        self.content_f_reg_model = content_f_reg_model
        self.cont_collab_mix = cont_collab_mix
        self.feature_compressor_content = Feature_compressor(n_features_content)
        self.feature_compressor_collab = Feature_compressor(n_features_collab)
        self.content_model = None
        self.collab_model = None        
        self.training_dataset = None
        
    def fit(self, X, y):   
        
        self.training_dataset = pd.concat([X, y], axis=1)
        features = X[self.feature_cols].drop(['Year'], axis=1)        
        self.feature_compressor_content.fit(features)
        self.feature_compressor_collab.fit(features)
        new_content_features = self.feature_compressor_content.transform(features)
        new_collab_features = self.feature_compressor_collab.transform(features)
        X_content = pd.concat([X[[self.user_col, self.item_col]], new_content_features, X['Year']], axis=1)
        X_collab = pd.concat([X[[self.user_col, self.item_col]], new_collab_features, X['Year']], axis=1)
        
        self.content_model = Content_filtering(
            user_col=self.user_col,
            item_col=self.item_col,
            rating_col=self.rating_col,
            feature_cols=X_content.drop([self.user_col, self.item_col], axis=1).columns,
            reg_model=self.content_f_reg_model
        )
        
        self.content_model.fit(X_content, y)
         
        self.collab_model = Collab_filtering(n_buddies=self.n_buddies, use_weights=False)
        self.collab_model.fit(X_collab, y)
        
    def predict(self, X):
        
        features = X[self.feature_cols].drop(['Year'], axis=1)
        new_content_features = self.feature_compressor_content.transform(features)
        new_collab_features = self.feature_compressor_collab.transform(features)
        X_content = pd.concat([X[[self.user_col, self.item_col]], new_content_features, X['Year']], axis=1)
        X_collab = pd.concat([X[[self.user_col, self.item_col]], new_collab_features, X['Year']], axis=1)
        y_pred_content = self.content_model.predict(X_content)        
        y_pred_collab = self.collab_model.predict(X_collab)        
        preds = (self.cont_collab_mix*y_pred_content + (1-self.cont_collab_mix)*y_pred_collab)
        return preds