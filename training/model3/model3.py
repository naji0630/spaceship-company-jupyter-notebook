!pip install lightgbm

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# 1. Load the data
train_df = pd.read_csv('train_with_gender.csv')
test_df = pd.read_csv('test_with_gender.csv')

# 2. Fill missing 'Age' values with the mean age from the training set
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(train_df['Age'].mean(), inplace=True)  # Use training mean for consistency

# 3. Split 'Cabin' into 'Deck', 'CabinNum', 'Side'
train_df[['Deck', 'CabinNum', 'Side']] = train_df['Cabin'].str.split('/', expand=True)
test_df[['Deck', 'CabinNum', 'Side']] = test_df['Cabin'].str.split('/', expand=True)

# 4. Drop unnecessary columns
train_df.drop(columns=['Name', 'FirstName', 'Cabin', 'CabinNum'], inplace=True)
test_df.drop(columns=['Name', 'FirstName', 'Cabin', 'CabinNum'], inplace=True)

# 5. One-hot encode categorical variables
categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'VIP', 'CryoSleep']
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# 6. Align the train and test datasets to have the same columns
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

missing_in_test = train_cols - test_cols
missing_in_train = test_cols - train_cols

# Add missing columns to test set
for col in missing_in_test:
    test_df[col] = 0

# Add missing columns to train set
for col in missing_in_train:
    train_df[col] = 0

# Ensure the order of columns in test set matches the train set
test_df = test_df[train_df.columns.drop('Transported')]

# 7. Define features and target variable
features = train_df.columns.drop(['Transported', 'PassengerId'])
X = train_df[features]
y = train_df['Transported'].astype(int)

# 8. Get PassengerId from test_df
passenger_ids = test_df['PassengerId']

# Drop 'PassengerId' from test_df
test_df = test_df.drop(columns=['PassengerId'])

# 9. Train the LightGBM classifier
model = LGBMClassifier(
    n_estimators=1768,
    max_depth=13,
    learning_rate=0.006759066453943107,
    num_leaves=36,
    subsample=0.8588280473652004,
    colsample_bytree=0.6947951715596529,
    random_state=42
)
model.fit(X, y)

# 10. Make predictions on the test set
test_predictions = model.predict(test_df)
test_predictions_bool = test_predictions.astype(bool)

# 11. Prepare the submission file
results_df = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Transported": test_predictions_bool
})
results_df.to_csv("submission.csv", index=False)

print("Saved 'submission.csv'")