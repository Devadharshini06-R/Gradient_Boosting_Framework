# Gradient_Boosting_Framework

Gradient boosting is a powerfull ML techniques, A step by step learning process where each model fixes the mistakes of the previous ones, producing a strong and accurate final model

visit my presentation [click here](https://www.pi.inc/docs/361585572091524?share_token=6SNHX5TQHBTCQ)

---

## Concept 

- Combining many weak learners to get a stronger predictive model
- Each new model attempts to correct the erroes made by the previous models

  ---
  ## Workflow

  - import libraries
  - Collect a dataset from sklearn.datasets
  - use Breast_cancer dataset for classification and California_housing for Regression
  - import necessary ML libraries
  - Handle Missing values and remove noice columns
  - Split the dataset into training and testing sets suing train_test_split
  - Importing Boosting model for both regression and classification
  - Using model and fit the train set for learning and predict the value by using testing set
  - comparing the values and calucate the performance matrices

  ---

  ### Load dataset

```
from sklearn.datasets import load_breast_cancer,fetch_california_housing
```
  ---
### Train_test_Split
```
from sklearn.model_selection import train_test_split
```

  ## Popular implementations

  - **XGBoost** 
  - **LightGBM**
  - **CatBoost**

  
  ```
  import xgboost as xgb
  import lightgbm as lgb
  from catboost import CatBoostRegressor
  from catboost import CatBoostClassifier
  ```
### model - XGBoost

```
from xgboost import XGBRegressor

# Model for XGBoost Regression
xgb_reg = XGBRegressor(
    n_estimators=100,   # Number of boosting rounds (trees)
    learning_rate=0.1,  # Step size shrinkage
    max_depth=3,        # Maximum depth of each tree
    random_state=42     # For reproducibility
)

xgb_reg.fit(X_train_housing, y_train_housing)



# Model for xgboost classification
xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_clf.fit(X_train_bcancer, y_train_bcancer)

```
--- 

## Model - LightGBM 

```
import lightgbm as lgb

lgb_reg = lgb.LGBMRegressor(
    n_estimators=200,     # Number of boosting rounds (trees)
    learning_rate=0.05,  # Shrinkage rate for each tree’s contribution
    max_depth=-1,        # No limit on depth (-1 means unlimited)
    num_leaves=31,       # Maximum leaves in one tree
    min_child_samples=5, # Minimum data needed in a leaf
    random_state=42,     # For reproducibility
    verbose=-1           # Suppress output logs
)

import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,   # Number of boosting rounds (trees)
    learning_rate=0.1,  # Step size shrinkage
    max_depth=3,        # Maximum depth of each tree
    random_state=42     # Ensures reproducibility
)

```
---

# Model - CatBoost

```
from catboost import CatBoostRegressor

cat_reg = CatBoostRegressor(
    iterations=100,    # Number of boosting rounds
    learning_rate=0.1, # Step size shrinkage
    depth=3,           # Depth of each tree
    verbose=0,         # Suppress training output
    random_state=42    # For reproducibility
)

cat_reg.fit(X_train_housing, y_train_housing)

from catboost import CatBoostClassifier

cat_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    verbose=0,
    random_state=42
)

cat_clf.fit(X_train_bcancer, y_train_bcancer)
```

