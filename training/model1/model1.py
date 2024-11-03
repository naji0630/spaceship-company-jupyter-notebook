import pandas as pd
from lightgbm import LGBMClassifier

train_df = pd.read_csv('../data/spaceship-titanic/train.csv')
test_df = pd.read_csv('../data/spaceship-titanic/test.csv')

passenger_ids = test_df['PassengerId']

train_df[['Deck', 'CabinNum', 'Side']] = train_df['Cabin'].str.split('/', expand=True)
train_df.drop(columns=['Name'], inplace=True)
train_df = pd.get_dummies(train_df, columns=['HomePlanet', 'Destination', 'Deck', 'Side', 'VIP', 'CryoSleep'], drop_first=True)

X = train_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'] + list(train_df.columns[-10:])]
y = train_df['Transported'].astype(int)

test_df[['Deck', 'CabinNum', 'Side']] = test_df['Cabin'].str.split('/', expand=True)
test_df.drop(columns=['Name'], inplace=True)
test_df = pd.get_dummies(test_df, columns=['HomePlanet', 'Destination', 'Deck', 'Side', 'VIP', 'CryoSleep'], drop_first=True)

for col in X.columns:
    if col not in test_df:
        test_df[col] = 0

test_df = test_df[X.columns]

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

test_predictions = model.predict(test_df)
test_predictions_bool = test_predictions.astype(bool)

results_df = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Transported": test_predictions_bool
})
results_df.to_csv("submission.csv", index=False)

print("submssion is saved.")
