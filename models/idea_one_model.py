# %%
import pandas as pd
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
df = pd.read_csv('data/idea_one/idea_one_data4.csv')
drop_cols = ['Crime_Occurances','Unnamed: 28']

# remove any underscores in column names
df.columns = df.columns.str.replace('_', '', regex=False)
df.drop(columns=drop_cols, inplace=True, errors='ignore')  # drop unnecessary columns
#df.drop('Unnamed: 28', axis=1, inplace=True)
df = df.dropna()

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
    5: (0.12 , float('inf')) # very high
}
affordability_bounds = {
    1: (0, 1307),   # very low
    2: (1307, 1404), # low
    3: (1404, 1509), # medium
    4: (1509, 1519), # high
    5: (1519, float('inf')) # very high
}

transit_bounds = {
    1: (0, 37),   # very low
    2: (37, 44), # low
    3: (44, 48), # medium
    4: (48, 56), # high
    5: (56, float('inf')) # very high
}

walkability_bounds = {  
    1: (0, 24),   # very low
    2: (24, 31), # low
    3: (31, 39), # medium
    4: (39, 55), # high
    5: (55, float('inf')) # very high
}

bikeability_bounds = {
    1: (0, 26),   # very low
    2: (26, 32), # low
    3: (32, 39), # medium
    4: (39, 47), # high
    5: (47, float('inf')) # very high
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


for key, value in user_input.items():
    if key in bounds_dict:
        lower_bound, upper_bound = bounds_dict[key][value]
        column_name = column_map[key]
        filtered_df = filtered_df[(filtered_df[column_name] >= lower_bound) & (filtered_df[column_name] <= upper_bound)]

bound_crime_occurances = [bounds_dict[key][value][0], bounds_dict[key][value][1]]
print(bound_crime_occurances)
bound_transit_scores = []
bound_walk_scores = []
bound_bike_scores = []
bound_affordability = []

print(filtered_df)

#%%%

X = df.drop(columns=['rent2024', 'rent2025','NeighbourhoodNumber', 'CenterLocation', 
                     'CityZone', 'CMHCZone', 'CrimeRate', 
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



# %%
