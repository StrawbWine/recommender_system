import pandas as pd
from sklearn.cluster import KMeans

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