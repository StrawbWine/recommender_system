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

from flask import Flask, request, jsonify, render_template
from waitress import serve

"""
Due to the high computational time (~3hrs) required to create a csv file with predictions for all possible combinations in the full dataset,
I will for this part just use the collaborative filtering part of my model.
This will lead to somewhat higher RMSE on predictions than would have been using the full model.
Another possibility is to load the 'predictions.csv' file here instead of model.pkl, which uses full power predictions, but only on the small dataset.

If you do have a couple of hours of compute time to spare, feel free to import the full dataset into the "predict.ipynb" file to generate a
csv with the complete predicting power of my full model.
All combos of BrukerID and FilmID are still included in this version, just with slightly less predictive power.

I apologize for putting the code for the classes right into app.py, but I just wanted to make sure I followed the file structure/format you wanted.
Ideally, the classes should of course reside in separate files and be imported.
"""

class Content_filtering:
    
    def __init__(
        self, 
        feature_cols,
        user_col='BrukerID',
        item_col ='FilmID',
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
        self.mean_rating = y.mean()
        
    def predict(self, X, return_as_array=True):        

        def try_predict(row):
            try:                
                return self.model[row[self.user_col]].predict(row[self.feature_cols].values.reshape(1, -1))[0]           
            except:                
                return pd.Series([self.mean_rating]*len(row))
            
        predictions = X.apply(try_predict, axis=1)        
        predictions = predictions.where(predictions <= 5.0, 5.0)        
        predictions = predictions.where(predictions >= 1.0, 1.0)                 
        if return_as_array:            
            return predictions.values
        else:            
            output_series = pd.Series(predictions.values)
            output_series.name = self.rating_col
            return output_series

class Collab_filtering:    
    
    def __init__(
        self,
        n_buddies=50,
        use_weights=False,
        user_col='BrukerID',
        item_col='FilmID',
        rating_col='Rangering'
    ):        
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

class Feature_compressor:
    
    def __init__(
        self,
        n_features
    ):
        self.n_features = n_features
        self.compression = None
        
    def fit(self, df):        
        corrs = df.corr()
        self.compression = KMeans(n_clusters=self.n_features, random_state=0).fit(corrs)
        
    def _combine_columns(self, df, columns, new_name, add_to_existing=False):
        df_new = df.iloc[:, columns].sum(axis=1)
        df_new = df_new.where(df_new == 0, 1)
        df_new.name = new_name
        if add_to_existing:
            df_new = pd.concat([df, df_new], axis=1)    
        return df_new    
        
    def transform(self, df):
        new_features = []
        for feature in range(self.n_features):
            indices = [i for i,x in enumerate(self.compression.labels_) if x == feature]
            new_feature = self._combine_columns(df, indices, str(feature))
            new_features.append(new_feature)
        new_features = pd.DataFrame(new_features).T
        return new_features

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



app = Flask(__name__)
model = pickle.load(open('C://Users\sever\INF161_final_submission/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():

    __input = dict(request.form)
    user_id = __input['BrukerID']
    film_id = __input['FilmID']    

    try:
        prediction = model.collab_model.model.loc[int(user_id), int(film_id)]
        if prediction > 5.0:
            prediction = 5.0
        elif prediction < 1.0:
            prediction = 1.0
        rounded_prediction = int(round(prediction))
    except:
        rounded_prediction = "Invalid input values...try again!"
        prediction = ""

    output_text = f'The algorithm predicts that user {user_id} will rate film {film_id}...:'

    return render_template('./index.html',
                            prediction_text=output_text,
                            prediction=prediction,
                            rounded_prediction=rounded_prediction)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)