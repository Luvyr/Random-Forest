import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Path of the file to read
iowa_file_path = './train.csv'

home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice

# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#creating model
rf_model = RandomForestRegressor(n_estimators = 300 , random_state =1)
rf_model.fit(train_X, train_y)
rf_predict = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_predict )
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(rf_val_mae))