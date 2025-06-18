# %%
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
# from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler


#%%
# get user input for their needs
user_safety_rating = int(input("How important is safety to you? (1-5): "))
# user_safety_rating = int(user_safety_rating)
if user_safety_rating < 1 or user_safety_rating > 5:
    raise ValueError("Safety rating must be between 1 and 5.")

user_affordability_rating = input("How important is affordability to you? (1-5): ")
user_affordability_rating = int(user_affordability_rating)
if user_affordability_rating < 1 or user_affordability_rating > 5:
    raise ValueError("Affordability rating must be between 1 and 5.")

user_tranist_rating = input("How important is transit to you? (1-5): ")
user_tranist_rating = int(user_tranist_rating)
if user_tranist_rating < 1 or user_tranist_rating > 5:
    raise ValueError("Transit rating must be between 1 and 5.")

user_walkability_rating = input("How important is walkability to you? (1-5): ")
user_walkability_rating = int(user_walkability_rating)
if user_walkability_rating < 1 or user_walkability_rating > 5:
    raise ValueError("Walkability rating must be between 1 and 5.")

user_bikeability_rating = input("How important is bikeability to you? (1-5): ")
user_bikeability_rating = int(user_bikeability_rating)
if user_bikeability_rating < 1 or user_bikeability_rating > 5:
    raise ValueError("Bikeability rating must be between 1 and 5.")

user_distance_uofa = input("How important is distance to U of A to you? (1-5): ")
user_distance_uofa = int(user_distance_uofa)
if user_distance_uofa < 1 or user_distance_uofa > 5:
    raise ValueError("Distance to U of A rating must be between 1 and 5.")

print(f"User ratings - Safety: {user_safety_rating}, Affordability: {user_affordability_rating}, Transit: {user_tranist_rating}, Walkability: {user_walkability_rating}, Bikeability: {user_bikeability_rating}, Distance to U of A: {user_distance_uofa}")

#%%
# load and preprocess your data
df = pd.read_csv('/Users/mylayambao/resiSense/data/idea_one/idea_one_data4.csv')
drop_cols = ['Crime_Occurances','Unnamed: 28']

# remove any underscores in column names
df.columns = df.columns.str.replace('_', '', regex=False)
#df.drop(columns=drop_cols, inplace=True, errors='ignore')  # drop unnecessary columns
df.drop('Unnamed: 28', axis=1, inplace=True)
df = df.dropna()

df = df[df['NeighbourhoodName'] != 'Calgary Trail North']


# %% Clustering Section
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Check that the columns exist
print("df columns:", df.columns.tolist())

# Select columns that are safe and valid for clustering
clustering_features = df[[
    'rent2019', 'rent2020', 'rent2021', 'rent2022', 'rent2023', 'CrimeRate',
    'WalkScore', 'TransitScore','BikeScore', 'Distance to U of A (km)', 'rent2024'
]]

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_features)

# Optional: Elbow method to decide best k
inertia = []
for k in range(1, 12):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Save elbow plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, 12), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.close()

# Fit final model with k=4 (or change this after checking elbow plot)
kmeans = KMeans(n_clusters=30, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(scaled_features)

# View clusters
print("\n Neighborhoods by Cluster:")
print(df[['NeighbourhoodName', 'Cluster']].drop_duplicates().sort_values('Cluster'))

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

# Save PCA cluster scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='Cluster', palette='tab10', s=80)
plt.title("Neighborhood Clusters (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("clusters.png")
plt.close()

print(df)

#Cluster interpretation summary
summary = df.groupby('Cluster')[[
   'rent2019', 'rent2020', 'rent2021', 'rent2022', 'rent2023', 'CrimeRate',
     'WalkScore', 'TransitScore','BikeScore', 'Distance to U of A (km)', 'rent2024'
]].mean().round(2)

print("\nCluster Summary:")
print(summary)

# Print sample neighborhoods from each cluster
for i in range(30):
    print(f"\nCluster {i} neighborhoods:")
    print(df[df['Cluster'] == i]['NeighbourhoodName'].unique()) 

# save the neighborhood names (as one col) with the cluster numbers (as another col)
neighborhoods_clusters = df[['NeighbourhoodName', 'Cluster']].drop_duplicates().sort_values('Cluster')
# Save the neighborhoods and clusters to a CSV file
neighborhoods_clusters.to_csv('neighborhoods_clusters.csv', index=False)
# Print confirmation
print("saved cluster csv")

#  Drop clustering columns before model training to avoid prediction errors
df.drop(columns=['Cluster', 'pca1', 'pca2'], inplace=True, errors='ignore')


#%% 
# save the clustering model
kmeans_model_path = 'kmeans_model.pkl'
import joblib
joblib.dump(kmeans, kmeans_model_path)
# Print confirmation
print(f"KMeans saved")

#%%
# labels = ['Very Low', 'Low', 'Neutral', 'High', 'Very High']
# df['Distance_UofA'], bins = pd.qcut(df['Distance to U of A (km)'], q=5, labels=labels, retbins=True)
# print(bins)

#%% 
# define upper and lower bounds for the saftey based on the user input

# input is a dictionary 
user_input = {
    'crimerate': user_safety_rating,
    'affordability': user_affordability_rating,
    'transitscore': user_tranist_rating,
    'walkscore': user_walkability_rating,
    'bikescore': user_bikeability_rating, 
    'uofa': user_distance_uofa
}
crime_rate_bounds = {
    1: (0.12 , 16.42),   # very low
    2: (0.07, 0.12), # low
    3: (0.04, 0.07), # medium
    4: (0.02, 0.04), # high
    5: (0, 0.02) # very high
}
affordability_bounds = {
    1: (1410, 1430),   # very lowAdd commentMore actions
    2: (1365, 1409),
    3: (1317, 1364), # medium
    4: (1129, 1316), # highAdd commentMore actions
    5: (1079, 1128)
}

transit_bounds = {
    1: (0, 37),   # very low
    2: (37, 44), # low
    3: (44, 48), # medium
    4: (48, 56), # high
    5: (56, 78) # very high
}

walkability_bounds = {  
    1: (0, 24),   # very low
    2: (24, 31), # low
    3: (31, 39), # medium
    4: (39, 55), # high
    5: (55, 89) # very high
}

bikeability_bounds = {
    1: (0, 26),   # very low
    2: (26, 32), # low
    3: (32, 39), # medium
    4: (39, 47), # high
    5: (47, 92) # very high
}

uofa_bounds = { 
    1: (11.73721675, 22.30119328),   # very low
    2: (9.71986469, 11.73721675), # low
    3: (7.45851460 , 9.71986468), # medium
    4: (4.92003644,  7.45851459 ), # high
    5: (0.2303176267, 4.92003643) # very high
}
# bounds dict
bounds_dict = {
    'crimerate': crime_rate_bounds,
    'affordability': affordability_bounds,
    'transitscore': transit_bounds,
    'walkscore': walkability_bounds,
    'bikescore': bikeability_bounds, 
    'uofa': uofa_bounds
}

column_map = {
    'crimerate': 'CrimeRate',
    'affordability': 'rent2024', 
    'transitscore': 'TransitScore', 
    'walkscore': 'WalkScore',
    'bikescore': 'BikeScore',
    'uofa': 'Distance to U of A (km)'
}

 #%%
def get_bounds(column_name, user_rating, reverse=False):
    col = column_map[column_name]
    rating = user_rating

    if rating not in bounds_dict[column_name]:
        raise ValueError(f"Invalid rating {rating} for {column_name}")

    if reverse:
        # HIGHER is better (walk/transit/bike scores)
        lower_vals = [bounds_dict[column_name][r][0] for r in bounds_dict[column_name] if r <= rating]
        min_val = bounds_dict[column_name][rating][0]
        max_val = max([bounds_dict[column_name][r][1] for r in bounds_dict[column_name]])
    else:
        # LOWER is better (crime, distance, affordability)
        lower_vals = [bounds_dict[column_name][r][0] for r in bounds_dict[column_name] if r >= rating]
        min_val = min(lower_vals)
        max_val = bounds_dict[column_name][rating][1]

    # median of selected category's range
    curr_range = bounds_dict[column_name][rating]
    #med_val = (curr_range[0] + curr_range[1]) / 2

    return [min_val, max_val]


#%%
# filter the dataframe based on user input
filtered_df = df.copy()

# drop unneeded columns
filtered_df = filtered_df.drop(columns=['NeighbourhoodNumber', 'CenterLocation', 
                                         'CityZone', 'CMHCZone', 'CrimeRate', 
                                         'SupportiveHousingCount', 'SupportiveUnits', 
                                         'SheltersCount',
                                        'Distance to MacEwan (km)', 
                                        'Distance to NAIT (km)', 
                                        'Distance to Concordia (km)',
                                        'Distance to NorQuest (km)','NeighbourhoodName', 'Population' ], axis=1)



bound_affordability = get_bounds('affordability', user_input['affordability'])
print(bound_affordability)
bound_crime_occurances = get_bounds('crimerate', user_input['crimerate'])
print(bound_crime_occurances)
bound_transit_scores = get_bounds('transitscore', user_input['transitscore'], reverse=True)
print(bound_transit_scores)
bound_walk_scores = get_bounds('walkscore', user_input['walkscore'],reverse=True)
print(bound_walk_scores)
bound_bike_scores = get_bounds('bikescore', user_input['bikescore'],reverse=True)
print(bound_bike_scores)
bound_uofa = get_bounds('uofa', user_input['uofa'])
print(bound_uofa)



#%%
# make a "hypothetical" dataframe with the combinations of the bounds lists

# group the historical values


# historical_affordability_bounds = list(zip(
#     bound_affordability2019,
#     bound_affordability2020,
#     bound_affordability2021,
#     bound_affordability2022
# ))

historical_affordability_bounds = []

for val in bound_affordability:
    rent_subset = filtered_df[filtered_df['rent2023'] >= val]
    if not rent_subset.empty:
        historical_affordability_bounds.append((
            rent_subset['rent2019'].max(),
            rent_subset['rent2020'].max(),
            rent_subset['rent2021'].max(),
            rent_subset['rent2022'].max()
        ))

# Other bounds stay the same
combo_lists = [
    historical_affordability_bounds,  # treated as 1 axis with tuples
    bound_affordability,
    bound_crime_occurances,
    bound_walk_scores,
    bound_transit_scores,
    bound_bike_scores, 
    bound_uofa
]

all_combinations = list(product(*combo_lists))

# flatten tuples
combo_df = pd.DataFrame(all_combinations, columns=[
    'historical_afford', 'rent2023', 'CrimeRate', 'WalkScore', 'TransitScore', 'BikeScore', 'Distance to U of A (km)'
])

# split historical affordability into separate columns
combo_df[['rent2019', 'rent2020', 'rent2021', 'rent2022']] = pd.DataFrame(combo_df['historical_afford'].tolist(), index=combo_df.index)

# drop the grouped column
combo_df = combo_df.drop(columns=['historical_afford'])

columns = [
    'rent2019', 'rent2020', 'rent2021', 'rent2022',
    'rent2023', 'CrimeRate', 'WalkScore', 'TransitScore', 'BikeScore', 'Distance to U of A (km)'
]
combo_df = combo_df[columns]
print(combo_df)




#%%%

X = df.drop(columns=['rent2024', 'rent2025','NeighbourhoodNumber', 'CenterLocation', 
                     'CityZone', 'CMHCZone', 'CrimeOccurances', 
                     'SupportiveHousingCount', 'SupportiveUnits', 
                     'SheltersCount',
                    'Distance to MacEwan (km)', 
                    'Distance to NAIT (km)', 
                    'Distance to Concordia (km)',
                    'Distance to NorQuest (km)', 'NewResidentialUnits2023','NumCurrentListings','Population' ], axis=1)
#print(X.dtypes)

y = df['rent2024']

# check the type of  y
print(type(df['rent2024']))
print(type(y[0]))

# convert the y to float
y = y.astype(float)
print(type(y[0]))

# Apply StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save NeighbourhoodName before dropping it
names = X_test['NeighbourhoodName'].values


# Drop NeighbourhoodName from X_train and X_test
X_test = X_test.drop('NeighbourhoodName', axis=1)
X_train = X_train.drop('NeighbourhoodName', axis=1)

print(X_train.dtypes)


#%%
model = MLPRegressor(random_state=1,max_iter=2000,tol=0.1, hidden_layer_sizes=5)
model.fit(X_train, y_train)

# %%
# evaluate the model 

y_pred = model.predict(X_test)

# metrics
score = model.score(X_test,y_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

#print(f'Score: {score:.4f}')
print(f'R-squared: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')


#%%
# TEST THE MODEL USING THE NEW COMBO DATAFRAME
y_pred_using_combo = model.predict(combo_df)


# print the columns and the predictions
# for i in range(len(y_pred_using_combo)):
#     print(f"hypothetical neighborhood {i+1}: {combo_df.iloc[i].to_dict()} -> rent estimate{y_pred_using_combo[i]:.2f}")

# %%
# store the hypothetical neighborhoods with rent estimates in a new dataframe
hypothetical_neighborhoods = combo_df.copy()
hypothetical_neighborhoods['rent2024'] = y_pred_using_combo
print(hypothetical_neighborhoods)


# %%
# use the kmeans saved kmeans model to predict the cluster for each hypothetical neighborhood
# load the saved model
kmeans = joblib.load('/Users/mylayambao/resiSense/models/kmeans_model.pkl')
# predict clusters for the hypothetical neighborhoods
hypothetical_neighborhoods_scaled = scaler.transform(hypothetical_neighborhoods)
print(hypothetical_neighborhoods_scaled)
hypothetical_neighborhoods['Cluster'] = kmeans.predict(hypothetical_neighborhoods_scaled)
# print the hypothetical neighborhoods with clusters
print("\nHypothetical Neighborhoods with Clusters:")
print(hypothetical_neighborhoods[['rent2024', 'Cluster']])
# %%
print(hypothetical_neighborhoods)

# %%

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(hypothetical_neighborhoods_scaled)
hypothetical_neighborhoods['pca1'] = pca_result[:, 0]
hypothetical_neighborhoods['pca2'] = pca_result[:, 1]

# Save PCA cluster scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=hypothetical_neighborhoods, x='pca1', y='pca2', hue='Cluster', palette='tab10', s=80)
plt.title("Neighborhood Clusters (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("clusters2.png")
plt.close()

# %%
# count  the number of neighborhoods in each cluster
cluster_counts = hypothetical_neighborhoods['Cluster'].value_counts().sort_index()
print("\nCluster Counts:")
print(cluster_counts)
# %%
# get all the real neighborhoods in the in the cluster of the 3 most reccomended clusters from the hypothetical neighborhoods

# %%
# get the top 3 recommended clusters
top_3_clusters = cluster_counts.nlargest(3).index.tolist()

# get the neighborhoods in those clusters
df_cluster = pd.read_csv('/Users/mylayambao/resiSense/neighborhoods_clusters.csv')
recommended_neighborhoods = df_cluster[df_cluster['Cluster'].isin(top_3_clusters)]

# assign recommendation scores based on cluster rank
recommendation_scores = {cluster: 3 - rank for rank, cluster in enumerate(top_3_clusters)}
recommended_neighborhoods['RecommendationScore'] = recommended_neighborhoods['Cluster'].map(recommendation_scores)

# Print the recommended neighborhoods
print("\nTop 3 Recommended Clusters and Neighborhoods:")
for cluster in top_3_clusters:
    print(f"\nCluster {cluster}:")
    neighborhoods_in_cluster = recommended_neighborhoods[recommended_neighborhoods['Cluster'] == cluster]['NeighbourhoodName'].tolist()
    for neighborhood in neighborhoods_in_cluster:
        print(neighborhood)

# Save the recommended neighborhoods to a CSV file with uppercase neighborhood names
recommended_neighborhoods['NeighbourhoodName'] = recommended_neighborhoods['NeighbourhoodName'].str.upper()
recommended_neighborhoods[['NeighbourhoodName', 'RecommendationScore']].to_csv('recommended_neighborhoods.csv', index=False)
