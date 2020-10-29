#!/usr/bin/env python
# coding: utf-8

# <h1>Cleaning datasets</h1>

# In[1]:


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


# <h2>Dataset "Bruker"</h2>

# I could easily figure out how to load the data correctly into a pandas dataframe by looking at the json file and comparing with the pandas read_json() documentation. 
# 
# Loading dataset and having an initial look at it:

# In[2]:


df_bruker = pd.read_json('C:\\Users\sever\OneDrive\Documents\kode\scripts\INF161\data/project/bruker.json', orient='split')
df_bruker.head()


# In[3]:


round(df_bruker.describe(),2)


# Check for duplicate user IDs:

# In[4]:


df_bruker['BrukerID'].drop_duplicates().shape[0] == df_bruker['BrukerID'].shape[0]


# The number of distinct rows in the 'BrukerID' column is the same as number of rows in the full column. Thus I can conclude that the all of the BrukerIDs are indeed unique.

# Investigate possible values for gender:

# In[5]:


df_bruker['Kjonn'].drop_duplicates()


# In[6]:


df_bruker['Kjonn'].value_counts()


# Investigate possible values for age:

# In[7]:


df_bruker['Alder'].value_counts()


# Investigating the values in the 'Postkode' column:

# In[8]:


df_bruker.loc[df_bruker['Postkode'].str.len() > 5].sort_values(by=['Postkode'], ascending=False)


# The 'Postkode' contains possibly different formats. Some domain knowledge about postal codes could be useful here to determine if some kind of cleanup of this column is necessary (for example by taking just the 5 first numbers of each row). As the different formats are also present in the clean dataset of the first part of the project, I'll just leave them as is for now.

# Checking how many rows of each column contains missing values:

# In[9]:


df_bruker.isna().sum()


# Three of the columns with missing data are nominal categorical data, and the last one is ordinal categorical data. If you are hellbent on completing these rows, you could apply multivariate imputation and look for correlations between the columns. For now I will not use these features in my machine learning models anyways, so I'll just simply replace the missing values with the most common value for each feature. Before directly using the imputer, I conform the missing value to np.nan.

# In[10]:


df_bruker = df_bruker.fillna(np.nan)
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df_bruker_cleaned = pd.DataFrame(imputer.fit_transform(df_bruker), columns=df_bruker.columns)
df_bruker_cleaned.head()


# Writing to disk in csv-format:

# In[11]:


df_bruker_cleaned.to_csv(
    path_or_buf='C:\\Users\sever\OneDrive\Documents\kode\scripts\INF161\data/project/cleaned/bruker.csv',
    index=False)


# <h2>Dataset "Film"</h2>

# Loading dataset and having an initial look at it:

# In[12]:


df_excel = pd.read_excel('C:\\Users\sever\OneDrive\Documents\kode\scripts\INF161\data/project/film.xlsx', sheet_name=None)
df_film = df_excel['film']
df_film.head()


# In[13]:


round(df_film.describe(),2)


# Checking for missing values:

# In[14]:


df_film.isna().sum()


# Investigating the values in the 'Sjanger' column. I want to be able to see a bit more than 10 rows/columns, so I change the display.max_rows/columns setting.

# In[15]:


pd.set_option("display.max_rows", 101)
pd.set_option("display.max_columns", 101)


# In[16]:


df_film['Sjanger'].value_counts().sort_values(ascending=False).head(20)


# I notice that the movies may have several genres associated with them. In these cases the genres are separated by a '|'. I want to have a look at how many genres a movie can have, could be useful information when I'm going to convert the column to a dummy variable format. 

# In[17]:


sjanger_counts = df_film['Sjanger'].str.count('\|').sort_values(ascending=False)
sjanger_counts


# Seems 6 is the most genres a single movie has in the dataset. Time to do conversion:
# 
# First I split the 'Sjanger' column on the '|' separator and expand it into dummy variable columns. I then create a copy of this dataframe.

# In[18]:


dfx = df_film['Sjanger'].str.split('|', expand=True)
dfx = pd.get_dummies(dfx)
dfx_2 = dfx.copy()


# For each genre combo, I will get a duplicate genre column. I want to merge these into one column for each genre, while keeping the information in the duplicate columns. To do this, I loop through the columns and for each column I loop through again to find other columns with identical suffixes (the duplicates will have names of the type "2_genre"). By only comparing the 5 first character of the genre names, I can also combine genres like "Children's" and "Children" into one. If I find a match, I will store the sum of columns as a new common column for this genre.

# In[19]:


for col_1 in dfx.columns:
    result = dfx[col_1].astype(int)    
    genre_1 = col_1.split('_')[1]
    for col_2 in dfx.columns:        
        genre_2 = col_2.split('_')[1]
        if genre_1[:5] == genre_2[:5]:
            result = result+dfx[col_2].astype(int)    
    dfx_2[genre_1] = result


# As the loop visited both "Children's" and "Children" I combine them here.
# 
# Right now, the new columns will have 0 for the movies where the genre is not represented, and some number > 1 for each row where the genre was represented. To convert this to the standard dummy variable format, I change all positive values to a 1. I can now drop the superfluous genre columns.

# In[20]:


dfx_2["Children's"] = dfx_2["Children"]+dfx_2["Children's"]    
dfx_2 = dfx_2.where(dfx_2 == 0, 1)
dfx_2 = dfx_2.drop(dfx_2.columns[:74], axis=1).drop(['Children'], axis=1)


# Finally, I will combine the cleaned genre columns with the rest of the columns from the original dataframe. I drop the duplicate index column and old 'Sjanger' column.
# 
# I notice a genre called "Ukjennt" which is present only in 3 rows, so I just drop this column too.

# In[21]:


df_film_cleaned = pd.concat([df_film, dfx_2], axis=1).drop(['Unnamed: 0', 'Sjanger'], axis=1)
df_film_cleaned = df_film_cleaned.loc[df_film_cleaned['Ukjennt'] != 1].drop(['Ukjennt'], axis=1)

df_film_cleaned.head()


# The dataset is now cleaned and converted to the correct format, so I can write it to disk as a csv file.

# In[22]:


df_film_cleaned.to_csv(
    path_or_buf='C:\\Users\sever\OneDrive\Documents\kode\scripts\INF161\data/project/cleaned/film.csv',
    index=False)


# <h2>Dataset "Rangering"</h2>

# Loading dataset and having an initial look at it:

# In[23]:


df_rang = pd.read_csv('C:\\Users\sever\OneDrive\Documents\kode\scripts\INF161\data/project/rangering.dat', sep='::', header=None)
df_rang.columns = ['BrukerID', 'FilmID', 'Rangering', 'Timestamp']
df_rang.head()


# In[24]:


df_rang.describe()


# Checking for missing values:

# In[25]:


df_rang.isna().sum()


# 1492 rows lack a timestamp. As I don't know which scale these values are on (pre/post 2000-08-01), I'll just drop these rows.

# In[26]:


df_rang = df_rang.dropna()


# Check that all values are between 1 and 10:

# In[27]:


df_rang['Rangering'].drop_duplicates().value_counts().sort_values(ascending=False)


# Ratings from before 2000-08-01 are on a scale of 1 to 10 while ratings after are on a scale from 1 to 5. We can convert the pre-change ratings to match the post-change ratings by halving the pre-change ratings.

# Converting date to unix time:

# In[28]:


change_time = (pd.Timestamp('2000-08-01 00:00:00') - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')


# In[29]:


pd.Timestamp(965088000, unit='s')


# Enforcing correct datatypes:

# In[30]:


df_rang['Timestamp'] = df_rang['Timestamp'].astype('int64')


# In[31]:


df_rang['Rangering'] = df_rang['Rangering'].astype('float64')


# In[32]:


df_rang.dtypes


# Separating pre and post change ratings, halving the pre change ratings and finally combining them again

# In[33]:


df_before_change = df_rang.loc[df_rang['Timestamp'] < change_time]
df_before_change_halved = df_before_change.copy()
df_before_change_halved['Rangering'] = df_before_change['Rangering']/2

df_after_change = df_rang.loc[df_rang['Timestamp'] >= change_time]

df_rang_cleaned = pd.concat([df_before_change_halved, df_after_change], axis=0)
df_rang_cleaned


# Dataset seems good to go, and I write it to disk as csv.

# In[34]:


df_rang_cleaned.to_csv(
    path_or_buf='C:\\Users\sever\OneDrive\Documents\kode\scripts\INF161\data/project/cleaned/rangering.csv',
    index=False)

