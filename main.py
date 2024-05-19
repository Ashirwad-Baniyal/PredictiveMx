import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
import joblib

# Load the dataset
df = pd.read_csv('predictive_maintenance.csv')
df.drop(['Product ID', "UDI", 'Target' , 'Type'], axis=1, inplace=True)

df["Failure Type"].replace({"No Failure": 0, "Heat Dissipation Failure": 1, "Power Failure": 2,
                             "Overstrain Failure": 3, "Tool Wear Failure": 4, "Random Failures": 5}, inplace=True)
# df["Type"].replace({"H": 0, "L": 1, "M": 2}, inplace=True)

predictive_columns = df.columns[:-1]
X = df[predictive_columns]
y = df["Failure Type"]

X_train, _, y_train, _ = train_test_split(
    X, y, random_state=40, test_size=0.33, stratify=y)

# Define preprocessing steps
categorical_cols = X.select_dtypes(include="object").columns.to_list()
categorical_pipe = Pipeline([('onehot', OneHotEncoder(sparse=False, handle_unknown="ignore"))])
numeric_pipe_1 = Pipeline([('power_transform', PowerTransformer())])
numeric_pipe_2 = Pipeline([('standardization', StandardScaler())]) 

# Full preprocessor
full = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipe, categorical_cols),
        ("power_transform", numeric_pipe_1, predictive_columns),
        ("standardization", numeric_pipe_2, predictive_columns)])

# CatBoost pipeline
catb = CatBoostClassifier(random_seed=42, logging_level='Silent')
pipeline_catb = Pipeline(steps=[("preprocess",full), ("base", catb)])

# Fit the model
model_catb = pipeline_catb.fit(X_train, y_train)

# Save the model
joblib.dump(model_catb, 'predictive_maintenance_model.pkl')
