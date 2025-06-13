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
user_safety_rating = input("How important is safety to you? (1-5): ")
user_safety_rating = int(user_safety_rating)
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

print(f"User ratings - Safety: {user_safety_rating}, Affordability: {user_affordability_rating}, Transit: {user_tranist_rating}, Walkability: {user_walkability_rating}, Bikeability: {user_bikeability_rating}")

#%%
# load and preprocess your data
df = pd.read_csv('/Users/mylayambao/resiSense/data/idea_one/idea_one_data4.csv')
drop_cols = ['Crime_Occurances','Unnamed: 28']

# remove any underscores in column names
df.columns = df.columns.str.replace('_', '', regex=False)
df.drop(columns=drop_cols, inplace=True, errors='ignore')  # drop unnecessary columns
#df.drop('Unnamed: 28', axis=1, inplace=True)
df = df.dropna()

#%%
# labels = ['Very Low', 'Low', 'Neutral', 'High', 'Very High']
# df['rent_2023_category'], bins = pd.qcut(df['rent2023'], q=5, labels=labels, retbins=True)
# print(bins)


#%% 
# define upper and lower bounds for the saftey based on the user input

# input is a dictionary 
user_input = {
    'crimerate': user_safety_rating,
    'affordability': user_affordability_rating,
    'transitscore': user_tranist_rating,
    'walkscore': user_walkability_rating,
    'bikescore': user_bikeability_rating
}
crime_rate_bounds = {
    1: (0, 0.02),   # very low
    2: (0.02, 0.04), # low
    3: (0.04, 0.07), # medium
    4: (0.07, 0.12), # high
    5: (0.12 , 16.42) # very high
}
affordability_bounds = {
    1: (1079, 1128),   # very low
    2: (1129, 1316), # low
    3: (1317, 1364), # medium
    4: (1365, 1409), # high
    5: (1410, 1430) # very high
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

# bounds dict
bounds_dict = {
    'crimerate': crime_rate_bounds,
    'affordability': affordability_bounds,
    'transitscore': transit_bounds,
    'walkscore': walkability_bounds,
    'bikescore': bikeability_bounds
}

column_map = {
    'crimerate': 'CrimeRate',
    'affordability': 'rent2024', 
    'transitscore': 'TransitScore', 
    'walkscore': 'WalkScore',
    'bikescore': 'BikeScore'
}

 #%%

# filter the dataframe based on user input
filtered_df = df.copy()

# drop unneeded columns
filtered_df = filtered_df.drop(columns=['NeighbourhoodNumber', 'CenterLocation', 
                                         'CityZone', 'CMHCZone', 'CrimeOccurances', 
                                         'SupportiveHousingCount', 'SupportiveUnits', 
                                         'SheltersCount','Distance to U of A (km)',
                                        'Distance to MacEwan (km)', 
                                        'Distance to NAIT (km)', 
                                        'Distance to Concordia (km)',
                                        'Distance to NorQuest (km)','NeighbourhoodName', 'Population', ], axis=1)



bound_affordability = list(bounds_dict['affordability'][user_input['affordability']])
bound_affordability2019 = [filtered_df[filtered_df['rent2023'] >= bound_affordability[0]]['rent2019'].max(), filtered_df[filtered_df['rent2023'] >= bound_affordability[1]]['rent2019'].max()]
bound_affordability2020 = [filtered_df[filtered_df['rent2023'] >= bound_affordability[0]]['rent2020'].max(), filtered_df[filtered_df['rent2023'] >= bound_affordability[1]]['rent2020'].max()]
bound_affordability2021 = [filtered_df[filtered_df['rent2023'] >= bound_affordability[0]]['rent2019'].max(), filtered_df[filtered_df['rent2023'] >= bound_affordability[1]]['rent2021'].max()]
bound_affordability2022 = [filtered_df[filtered_df['rent2023'] >= bound_affordability[0]]['rent2022'].max(), filtered_df[filtered_df['rent2023'] >= bound_affordability[1]]['rent2022'].max()]
bound_crime_occurances = list(bounds_dict['crimerate'][user_input['crimerate']])
bound_transit_scores = list(bounds_dict['transitscore'][user_input['transitscore']])
bound_walk_scores = list(bounds_dict['walkscore'][user_input['walkscore']])
bound_bike_scores = list(bounds_dict['bikescore'][user_input['bikescore']])

#%%
# make a "hypothetical" dataframe with the combinations of the bounds lists
# the order for each row should be bound_affordability2019, bound_affordability2020, bound_affordability2021, bound_affordability2022, bound_affordability ,bound_crime_occurances, bound_walk_scores, bound_transit_scores, bound_bike_scores
combo_lists = [
    bound_affordability2019,
    bound_affordability2020,
    bound_affordability2021,
    bound_affordability2022,
    bound_affordability,
    bound_crime_occurances,
    bound_walk_scores,
    bound_transit_scores,
    bound_bike_scores
]

all_combinations = list(product(*combo_lists))

columns = [
    'rent2019', 'rent2020', 'rent2021', 'rent2022',
    'rent2023', 'CrimeRate', 'WalkScore', 'TransitScore', 'BikeScore'
]
combo_df = pd.DataFrame(all_combinations, columns=columns)
print(combo_df)






#%%%

X = df.drop(columns=['rent2024', 'rent2025','NeighbourhoodNumber', 'CenterLocation', 
                     'CityZone', 'CMHCZone', 'CrimeOccurances', 
                     'SupportiveHousingCount', 'SupportiveUnits', 
                     'SheltersCount','Distance to U of A (km)',
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

for i in range(len(y_pred)):
    print(f"{names[i]}: predicted: {y_pred[i]:.2f}  actual: {y_test.iloc[i]:.2f}")

for i in range(len(y_pred)):
    diff = y_pred[i]- y_test.iloc[i]
    print(f"{names[i]}: difference: {diff:.2f}")

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

print(y_pred_using_combo)

# print the columns and the predictions
for i in range(len(y_pred_using_combo)):
    print(f"hypothetical neighborhood {i+1}: {combo_df.iloc[i].to_dict()} -> rent estimate{y_pred_using_combo[i]:.2f}")

# %%
