#!/usr/bin/env python
# coding: utf-8

# In[47]:


import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[48]:


data = pd.read_csv("spotify.csv")


# In[49]:


print(data.head)
print(data.info)


# In[50]:


df = pd.DataFrame(data)
print(data.song_name)
# Map song IDs to their corresponding names
df["song_nam"] = df['song_id'].map(data.song_name)

# Display the DataFrame
print("Sample of the dataset:")
print(df.head())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Filter data based on condition (repeated_plays = 1)
repeated_plays_df = df[df['Plays within a Month'] == 1]
print("\nData where repeated_plays is 1:")
print(repeated_plays_df)



# In[54]:


plt.figure(figsize=(10, 6))
repeated_plays_df['song_name'].value_counts().plot(kind='bar', color='orange')
plt.title('Songs with Repeated Plays')
plt.xlabel('Song Name')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[55]:


X = pd.get_dummies(data.drop(['Plays within a Month'], axis=1))
y = data['Plays within a Month']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)


# In[56]:


# Make predictions on the test set
predictions = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# In[ ]:




