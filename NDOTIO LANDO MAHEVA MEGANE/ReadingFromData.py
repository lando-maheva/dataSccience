import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
#
# file_path = './Iowa/IowaHousing.csv'
#
home_data = pd.read_csv('./Iowa/IowaHousing.csv')
y = home_data.SalePrice
# #we read the data
# home_data = pd.read_csv(file_path)
#
# # to view summary statistics we use the describe function
# # print(home_data.describe())
#
# # answers based on Ames column names
# avg_lot_size = round(home_data['Lot Area'].mean())
# newest_home_age = 2026 - home_data['Year Built'].max()
# #
# # print(f"\nAverage lot size: {avg_lot_size}")
# # print(f"Age of the newest home: {newest_home_age}")
#
# newest_build_year = home_data['Year Built'].max()
# newest_home_age = 2026 - newest_build_year
# print(f"The newest home was built in {newest_build_year}.")
# print(f"As of today (2026), the newest home is {newest_home_age} years old.")
# #
#
# # # we save the sales price to the variable y
# y = home_data.SalePrice
# #
# # # we then display the top few rows to verify
# # # ?]
# #
feature_names = [
    'Lot Area',
    'Year Built',
    '1st Flr SF',
    '2nd Flr SF',
    'Full Bath',
    'Bedroom AbvGr',
    'TotRms AbvGrd'
]
X = home_data[feature_names]
# # #we prnt the result
# #
# # print("First 5 rows of X:")
# # print(X.head())
# #
# # print("\nSummary statistics for X:")
# # print(X.describe())
iowa_model = DecisionTreeRegressor(random_state=1)
#
# # The model maps the features (X) to the target prices (y).
iowa_model.fit(X, y)
#
# print("Model training complete.")
predictions = iowa_model.predict(X)
#
# # printing the first 5 predicted prices
print("First 5 predictions:")
print(predictions[:5])
#
# # Compare predictions to actual values
print("Actual target values (y):")
print(y.head().tolist())
#
# print("\nModel predictions:")
# print(predictions[:5])

from sklearn.model_selection import train_test_split
#
# # splits data into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
iowa_model = DecisionTreeRegressor(random_state=1)
#
# # fit the model with the trained data
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
#
#)# printing the result
print(f"Validation MAE: ${val_mae:,.2f}")

# using random_state=1 ensures your results match the expected output exactly
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
print(" the model is specified and fitted to the training data")

val_predictions = iowa_model.predict(val_X)
print("the top 5 Validation Predictions include:")
print(val_predictions[:5])
print("\nthe top 5 Actual Prices (Validation Data) are :")
print(val_y.head())

#
# def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
#     model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(train_X, train_y)
#     preds_val = model.predict(val_X)
#     mae = mean_absolute_error(val_y, preds_val)
#
#     return mae
# # test 5 leaves
# mae_5 = get_mae(5, train_X, val_X, train_y, val_y)
# # test 50 leaves
# mae_50 = get_mae(50, train_X, val_X, train_y, val_y)
# # test 100 leaves
# mae_100 = get_mae(100, train_X, val_X, train_y, val_y)
# # test 500 leaves
# mae_500 = get_mae(500, train_X, val_X, train_y, val_y)
# #print all the leaves and compare
# print(f"MAE for 5 leaves:   ${mae_5:,.0f}")
# print(f"MAE for 50 leaves:  ${mae_50:,.0f}")
# print(f"MAE for 100 leaves: ${mae_100:,.0f}")
# print(f"MAE for 500 leaves: ${mae_500:,.0f}")

# best_tree_size = 100
# final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
# final_model.fit(X, y)
# print("final model is trained on all data")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_val_predictions)

print(f"the validation mean average error  for Random Forest: ${rf_val_mae:,.2f}")