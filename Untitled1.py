#!/usr/bin/env python
# coding: utf-8
#Importing Ibraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#DOWNLOADING DATASETS
df_movie=pd.read_csv('C:\\Users\\satya\\OneDrive\\Desktop\\movies.dat',sep="::",engine='python', encoding='unicode_escape')
df_movie.dropna(inplace=True)
df_movie.head()


# In[50]:


df_movie.shape


# In[51]:


df_movie.describe()


# In[52]:


df_movie.isna().sum()

# DOWNLOADING DATASETS ratings_data

# In[57]:


df_ratings=pd.read_csv('C:\\Users\\satya\\OneDrive\\Desktop\\ratings.dat', sep="::", engine="python")
df_ratings.dropna(inplace=True)
df_ratings.head(10)


# In[59]:


df_ratings.shape


# In[33]:


ratings_df


# In[61]:


df_ratings.isna().sum()

DOWNLOADING DATASETS Users_data

# In[62]:


df_users=pd.read_csv('C:\\Users\\satya\\OneDrive\\Desktop\\users.dat', sep="::", engine="python")
df_users.dropna(inplace=True)
df_users.head(10)


# In[63]:


df_users.shape


# In[64]:


df_users.describe()


# In[69]:
# Create a new dataset [Master_Data] with the following columns MovieID Title UserID Age Gender Occupation Rating. 
#  (i) Merge two tables at a time. (ii) Merge the tables using two primary keys MovieID & UserId
movie_ratings_df = movies_df.merge(ratings_df, left_on='Id', right_on='MovieId')
master_df = movie_ratings_df.merge(users_df,left_on='UserId', right_on='Id')


# In[97]:
master_df = master_df[['Name','Genre','UserId','MovieId','Rating','TimeStamp','Gender','Age Range','Occupation Category','ZipCode']]
master_df

# In[71]:
master_df.shape

# In[78]:
df2=master_df.drop(["Occupation Category","ZipCode","TimeStamp"],axis=1)
df2.head()

# In[79]:
df2.describe()

# In[80]:
df2.isna().sum()

# In[81]:
df_final=df2.dropna()

# In[82]:
df_final.shape

# VISULAZING THE PREPROCESSED DATA
# In[84]:
sns.countplot(x=df_final['Gender'],hue=df_final['Rating'])


# In[102]:
plt.figure()
users_df['Age Range'].hist()
plt.title('User Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.show()


# In[104]:
df_final['Rating'].value_counts().plot(kind='bar')
plt.show()


# In[110]:
df_final['MovieId'].plot.hist(bins=25)
plt.xlabel("MovieId")
plt.ylabel("Ratings")


# In[112]:
df_final['Age Range'].plot.hist(bins=10)


# In[115]:
sns.countplot(x=df_final['Age Range'],hue=df_final['Rating'])


# In[116]:
df_final.head()


# In[118]:
input=df_final.drop(['Rating','Name','Genre','MovieId'], axis=1)
target=df_final['Rating']


# In[119]:
target.head()


# In[120]:
input.head()


# In[129]:
#Top 25 movies by viewership rating
moviesByRating = master_df.groupby(['Name'])['Rating'].mean()
moviesByRating.sort_values(ascending=False).head(25)


# In[123]:
#Most rated movies 
moviesByRatingCount = master_df.groupby(['Name'])['Rating'].count()
moviesByRatingCount.sort_values(ascending=False).head(25)


# In[128]:
#Find the ratings for all the movies reviewed by for a particular user of user id = 2696
ratingsByUser = master_df[master_df['UserId'] == 2696].sort_values(by='Rating',ascending=False)
print('Number of ratings by userId 2696:', ratingsByUser['UserId'].count())
ratingsByUser


# In[130]:
genre_df = master_df['Genre'].drop_duplicates().str.split('|',expand=True)
oneCol = []
for x in range(len(genre_df.columns)):
    oneCol.append(genre_df[x])
uniqueGenres = pd.concat(oneCol, ignore_index=True)
uniqueGenres = uniqueGenres.dropna().unique()
uniqueGenres


# In[131]:
# Convert Categorical to numeric data.
Genre = master_df['Genre'].str.get_dummies().add_prefix('Genres_')
Genre


# In[132]:
# Drop the existing Genre (Categorical) and add the new Genre(numeric data) to the master_df.
master_df = pd.concat(
    [master_df.drop(
        ['Genre'],
        axis=1
    ),
     Genre],
    axis=1,
   
)


# In[133]:
# Convert Gender from categorical M/F to numeric 0/1.
master_df = pd.get_dummies(
    master_df,
    columns=['Gender']
)


# In[134]:
# Show all columns
pd.set_option('display.max_columns', None)

master_df.head()


# In[135]:
# Find the correlation for Rating.
correlation_matrix = master_df.corr()
correlation_matrix['Rating']


# In[136]:
master_df.columns


# In[155]:
# Build a Model to predict user rating : 
# Define X (independent variables) and Y (dependent variable)
X = master_df[['Age Range', 'Occupation Category',
       'Genres_Action', 'Genres_Adventure', 'Genres_Animation',
       'Genres_Children\'s', 'Genres_Comedy', 'Genres_Crime',
       'Genres_Documentary', 'Genres_Drama', 'Genres_Fantasy',
       'Genres_Film-Noir', 'Genres_Horror', 'Genres_Musical', 'Genres_Mystery',
       'Genres_Romance', 'Genres_Sci-Fi', 'Genres_Thriller', 'Genres_War',
       'Genres_Western', 'Gender_F', 'Gender_M']]
y = master_df['Rating']


# In[156]:
X


# In[158]:
X.shape


# In[159]:
# 'random' split data in train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state= 100)

# 80% in train
# 20% data in test


# In[160]:
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[162]:
y_train.head()


# In[163]:
X_test.head()


# In[164]:
# Check if All dtypes are numeric, so they can be processed by ml algorithm.
X_train.dtypes


# In[166]:
y_test.head()


# In[167]:
# Rating is categorical data with values 1,2,3,4,5. So, we use Logistic Regression.
from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()
logreg = LogisticRegression(max_iter=100000)

lm.fit(X_train, y_train)   # training


# In[168]:
# predict the movie rating on test data
prediction = lm.predict(X_test)


# In[169]:
prediction


# In[170]:
y_test


# In[171]:
test=pd.DataFrame({'Predicted':prediction, 'Actual': y_test})


# In[172]:
#Print the accuarcy score.
from sklearn import metrics
metrics.accuracy_score(y_test,prediction)


# In[173]:
test.head(40)

