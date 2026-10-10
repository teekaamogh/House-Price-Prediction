import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the data
housing = pd.read_csv("housing.csv")

# 2. New column with range of income
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5]) 

# 3. Stratified shuffle split 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
  strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
  strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)

# 3. Work on the copy of training data
housing = strat_train_set.copy()

# 4. Seperate features and labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

# 5. Seperate categorial and numerical columns
num_attribs = housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_attribs = ['ocean_proximity']

# 6. Making pipeline 

# For numerical columns
num_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy='median')),
  ('standard', StandardScaler())
])

# For categorical columns
cat_pipeline = Pipeline([
  ('onehot', OneHotEncoder(handle_unknown="ignore"))  # Suppose there are 5 types in ocean_proximity and when new type is added then it ignores
])

# Construct the full pipeline
full_pipeline = ColumnTransformer([
  ('num', num_pipeline, num_attribs),
  ('cat', cat_pipeline, cat_attribs)
])

# 7. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

# 8. Train the model

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Decision tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

# Random Forest
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# Predict using training data
lin_preds = lin_reg.predict(housing_prepared)
tree_preds = tree_reg.predict(housing_prepared)
forest_preds = forest_reg.predict(housing_prepared)

# Calculate RMSE
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
tree_rmse = root_mean_squared_error(housing_labels, tree_preds)
forest_rmse = root_mean_squared_error(housing_labels, forest_preds)

print("Linear Regression RMSE:", lin_rmse)
print("Decision Tree RMSE:", tree_rmse)
print("Random Forest RMSE:", forest_rmse)

# Evaluate Decision Tree with cross-validation
tree_rmse_cv = -cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print("Decision Tree RMSE after cross validation:",tree_rmse_cv.mean())
