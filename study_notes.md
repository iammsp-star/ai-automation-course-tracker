# AI & Automation Course Notes & Summaries

Track: Cognitive Class by IBM
Total Target: 120 Hours

---

## Kaggle Learn: Intro to Machine Learning

### 1. Overfitting vs. Underfitting

- **Tree Depth & Model Complexity**:
  - A Decision Tree splits data based on feature thresholds.
  - A shallow tree (e.g., depth 1 or 2, 2 to 4 leaves) groups many diverse observations together.
  - A deep tree (e.g., depth 10, up to $2^{10} = 1024$ leaves) slices data into tiny subsets containing very few observations per leaf.

- **Overfitting**:
  - Occurs when a model learns training data too precisely, capturing noise and spurious patterns that do not generalize to unseen data.
  - **Symptom**: Extremely low error on training data, but high Mean Absolute Error (MAE) on validation/test data.
  - **Cause**: Deep trees with too many leaf nodes.

- **Underfitting**:
  - Occurs when a model is overly simplistic and fails to capture the underlying structure and relationships in data.
  - **Symptom**: High MAE on both training data and validation data.
  - **Cause**: Shallow trees with too few leaf nodes.

- **The Sweet Spot (Optimal Model Complexity)**:
  - The balance point where model complexity minimizes the validation MAE (the lowest point of the validation error curve).

---

### 2. Hyperparameter Tuning with `max_leaf_nodes`

- Scikit-learn's `DecisionTreeRegressor` allows controlling tree complexity via `max_leaf_nodes`.
- A utility evaluation function measures validation MAE across candidate values of `max_leaf_nodes` (e.g., `[5, 50, 500, 5000]`).
- Example evaluation on Melbourne housing dataset:
  - `max_leaf_nodes=5`: MAE = 347,380 (Underfitting)
  - `max_leaf_nodes=50`: MAE = 258,171
  - `max_leaf_nodes=500`: MAE = 243,495 (Optimal balance point)
  - `max_leaf_nodes=5000`: MAE = 254,983 (Overfitting)

---

### 3. Ensemble Models & Random Forests

- **Ensemble Models**:
  - Machine learning methods that construct and combine multiple base estimators (e.g., individual decision trees) to produce a single aggregated prediction that is superior to any single model.

- **Random Forest Architecture**:
  - Constructs a "forest" of numerous decision trees, each trained on randomized bootstrap samples of the dataset and subsets of features.
  - **Prediction Mechanism**: Obtains predictions from each component tree and averages them (`RandomForestRegressor`) or uses majority vote (`RandomForestClassifier`).

---

### 4. Key Advantages of `RandomForestRegressor`

1. **Superior Accuracy**: Averaging across multiple decision trees drastically reduces variance without increasing bias, producing significantly lower validation MAE (e.g., MAE drops from ~243,495 with tuned Decision Tree to ~191,669 with Random Forest).
2. **Robustness to Overfitting**: By aggregating randomized trees, random forests resist overfitting far better than individual deep decision trees.
3. **Out-of-the-Box Performance**: Works exceptionally well using standard default parameters (e.g., `RandomForestRegressor(random_state=1)`), requiring far less manual hyperparameter tuning compared to single trees or gradient boosted models.

---

### 5. Executable Python Snippets

#### Snippet A: Utility Function for `max_leaf_nodes` Tuning
```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """
    Fits a DecisionTreeRegressor with specified max_leaf_nodes
    and returns validation Mean Absolute Error (MAE).
    """
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae
```

#### Snippet B: Complete Reproducible Demonstration (Decision Tree Tuning vs. Random Forest)
```python
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)

# 1. Generate synthetic regression dataset representing tabular feature data
X_raw, y_raw = make_regression(
    n_samples=1000, 
    n_features=10, 
    noise=20.0, 
    random_state=42
)
X = pd.DataFrame(X_raw, columns=[f'feature_{i}' for i in range(10)])
y = pd.Series(y_raw, name='target')

# 2. Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# 3. Tune max_leaf_nodes for DecisionTreeRegressor
candidate_max_leaf_nodes = [5, 50, 500, 5000]
print("--- Decision Tree Hyperparameter Tuning ---")
scores = {}
for max_leaves in candidate_max_leaf_nodes:
    mae = get_mae(max_leaves, train_X, val_X, train_y, val_y)
    scores[max_leaves] = mae
    print(f"Max leaf nodes: {max_leaves:4d} | Validation MAE: {mae:.2f}")

best_tree_leaves = min(scores, key=scores.get)
print(f"Optimal max_leaf_nodes for Decision Tree: {best_tree_leaves} (MAE: {scores[best_tree_leaves]:.2f})")

# 4. Train RandomForestRegressor
print("\n--- Random Forest Regressor ---")
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
rf_preds = forest_model.predict(val_X)
rf_mae = mean_absolute_error(val_y, rf_preds)
print(f"RandomForestRegressor Validation MAE: {rf_mae:.2f}")

# 5. Model Comparison Summary
improvement = ((scores[best_tree_leaves] - rf_mae) / scores[best_tree_leaves]) * 100
print(f"\nRandom Forest reduced MAE by {improvement:.2f}% relative to the best Decision Tree!")
```

---

## Kaggle Learn: How Models Work & Basic Data Exploration

### 1. How Models Work
- **Decision Trees**:
  - The fundamental building block of machine learning models for tabular data.
  - Divides observations into categories using sequential conditional splits based on input features (e.g., *Is Lot Size > 1000 sq ft?*).
  - **Splits**: Decision nodes where the dataset is partitioned based on feature value thresholds.
  - **Leaves (Terminal Nodes)**: Bottom points of the tree where predictions are made. The predicted value for any observation reaching a leaf is the historical average target value of training samples in that leaf.
  - **Tree Depth**: Number of split levels from root to leaf. Deeper trees evaluate combinations of multiple features simultaneously (e.g., bedroom count + lot size).
- **Fitting / Training**:
  - The process of capturing patterns from data.
  - Data used to train/fit the model is called **training data**.
  - During fitting, the model algorithm automatically determines optimal split criteria and leaf values.
- **Target Variable ($y$)**:
  - The column or quantity you want to predict (e.g., house price `Price`).
- **Feature Selection ($X$)**:
  - The selected set of input variables used by the model to predict the target variable (e.g., `Rooms`, `Bathroom`, `Landsize`, `Lattitude`, `Longtitude`).

---

### 2. Basic Data Exploration with Pandas
- **Pandas Library**: The essential Python data science library for data manipulation and analysis (`import pandas as pd`).
- **Pandas DataFrame**: A 2-dimensional labeled data structure holding data in rows and columns, similar to a SQL table or Excel spreadsheet.
- **`DataFrame.describe()`**: Core Pandas method that returns summary statistics for numerical columns in a DataFrame.
- **Summary Statistics Breakdown**:
  - **`count`**: Number of non-missing (non-null) values per column. Missing values arise when data is uncollected or non-applicable (e.g., 2nd bedroom area in 1-bedroom houses).
  - **`mean`**: The arithmetic average value.
  - **`std`**: Standard deviation, quantifying the spread/dispersion of values around the mean.
  - **`min`**: Minimum value observed in the column.
  - **Percentiles (`25%`, `50%`, `75%`)**:
    - **`25%` (25th Percentile / 1st Quartile)**: Value greater than 25% of data points and less than 75%.
    - **`50%` (50th Percentile / Median / 2nd Quartile)**: Middle value dividing the sorted column values in half.
    - **`75%` (75th Percentile / 3rd Quartile)**: Value greater than 75% of data points.
  - **`max`**: Maximum value observed in the column.

---

### 3. Code Examples & Exploratory Workflow

#### Snippet C: Data Loading and Summarization
```python
import pandas as pd

# Load dataset into Pandas DataFrame
file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(file_path)

# Display numerical summary statistics
print(melbourne_data.describe())
```

#### Snippet D: Target Variable & Feature Matrix Selection
```python
import pandas as pd

# 1. Load dataset
melbourne_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# 2. Select Target Variable (y)
y = melbourne_data.Price

# 3. Select Features (X)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# 4. Inspect Features Summary Statistics & Top Rows
print("--- Feature Matrix Summary Statistics ---")
print(X.describe())

print("\n--- Top 5 Rows of Feature Matrix ---")
print(X.head())
```

---

## Kaggle Learn: Your First Machine Learning Model & Model Validation

### 4. Building Your First Machine Learning Model

- **Scikit-learn (`sklearn`)**: The standard Python machine learning library (`from sklearn.tree import DecisionTreeRegressor`).
- **4 Steps of Model Building**:
  1. **Define**: Select the algorithm type (e.g., `DecisionTreeRegressor`) and specify parameters such as `random_state=1` for reproducibility.
  2. **Fit**: Pass features `X` and target `y` to `.fit(X, y)` to capture dataset patterns.
  3. **Predict**: Use `.predict(X)` to generate output values.
  4. **Evaluate**: Assess the accuracy of the predictions.

- **Prediction Target (`y`) vs. Features (`X`)**:
  - **Target (`y`)**: The single variable to predict (Pandas Series).
  - **Features (`X`)**: Input attributes used by the model to calculate predictions (Pandas DataFrame).

---

### 5. Model Validation & Train-Test Split

- **What is Model Validation?**:
  - Evaluating model predictive accuracy on unseen data to test quality and select optimal models.

- **The Fallacy of In-Sample Evaluation**:
  - Calculating model performance on the training data used to fit it produces an **in-sample score**.
  - **Pitfall**: Models easily fit noise or false correlations in training data (e.g., linking door color to house price).
  - **Consequence**: In-sample MAE appears artificially low (~$0 to $500), but out-of-sample error on real unseen data is massive ($250,000+).

- **Train-Test Split (`train_test_split`)**:
  - Divides data into:
    - **Training Set (`train_X`, `train_y`)**: Used exclusively to fit the model.
    - **Validation Set (`val_X`, `val_y`)**: Held out during training and used solely for prediction evaluation.
  - Implemented via `from sklearn.model_selection import train_test_split`.

- **Mean Absolute Error (MAE)**:
  - Metric definition: $\text{MAE} = \text{mean}(|\text{actual} - \text{predicted}|)$.
  - Calculated via `from sklearn.metrics import mean_absolute_error`.
  - Plain English interpretation: "On average, our model predictions are off by $X$."

---

### 6. Executable Python Snippet: First Model & Validation Demonstration

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Generate synthetic housing dataset
np.random.seed(42)
n_samples = 500
X = pd.DataFrame({
    'Rooms': np.random.randint(1, 6, n_samples),
    'Bathroom': np.random.randint(1, 4, n_samples),
    'Landsize': np.random.randint(100, 1000, n_samples),
    'Lattitude': np.random.uniform(-38.0, -37.7, n_samples),
    'Longtitude': np.random.uniform(144.9, 145.1, n_samples)
})
y = X['Rooms'] * 100000 + X['Bathroom'] * 50000 + X['Landsize'] * 200 + np.random.normal(0, 15000, n_samples)

# 2. In-Sample Evaluation (Flawed)
insample_model = DecisionTreeRegressor(random_state=1)
insample_model.fit(X, y)
insample_preds = insample_model.predict(X)
insample_mae = mean_absolute_error(y, insample_preds)

print("--- 1. In-Sample Evaluation (Flawed) ---")
print(f"In-Sample MAE: ${insample_mae:.2f}")

# 3. Validation Set Evaluation using train_test_split (Correct)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

val_model = DecisionTreeRegressor(random_state=1)
val_model.fit(train_X, train_y)
val_preds = val_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_preds)

print("\n--- 2. Validation Set Evaluation (Correct) ---")
print(f"Validation MAE: ${val_mae:.2f}")
print(f"Out-of-sample Error Discrepancy: ${val_mae - insample_mae:.2f}")
```
