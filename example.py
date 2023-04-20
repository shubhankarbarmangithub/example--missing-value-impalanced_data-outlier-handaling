import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('dataset.csv')

# Split the dataset into features and target
X = df.drop('target_col', axis=1)
y = df['target_col']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Replace missing values in categorical columns with mode (on the training set only)
cat_cols = ['cat_col', 'date_col']
for col in cat_cols:
    mode = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(mode)
    X_test[col] = X_test[col].fillna(mode)

# Replace missing values in integer columns with median (on the training set only)
num_cols = ['int_col']
for col in num_cols:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)

# Calculate the VIF for each feature
vif = pd.DataFrame()
vif["feature"] = X_train.columns
vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# Drop the features with high VIF values
X_train = X_train.drop(vif[vif['VIF'] > 5].index, axis=1)
X_test = X_test[X_train.columns]

# Remove outliers from the training set using IQR method (on the training set only)
num_cols = ['int_col']
for col in num_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    X_train = X_train[(X_train[col] >= Q1 - 1.5*IQR) & (X_train[col] <= Q3 + 1.5*IQR)]
    y_train = y_train[X_train.index]

# Apply SMOTE to the training set
smote = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize the features
sc = StandardScaler()
X_train_resampled = sc.fit_transform(X_train_resampled)
X_test = sc.transform(X_test)

# Create the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the model
model.fit(X_train_resampled, y_train_resampled)

# Predict the target values
y_pred = model.predict(X_test)

# Print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
