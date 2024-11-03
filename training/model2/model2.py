# 1. 데이터 로드 및 전처리
train_df = pd.read_csv('./data/spaceship-titanic/train.csv')
test_df = pd.read_csv('./data/spaceship-titanic/test.csv')

# PassengerId 열 유지
passenger_ids = test_df['PassengerId']

# 2. Gender 모델 불러오기
with open('gender_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

def extract_first_name(full_name):
    if pd.isna(full_name):
        return None
    return full_name.split()[0]  # 첫 번째 단어를 반환 (이름)

def predict_gender_probability(name):
    if not name:  # 이름이 없는 경우
        return 0.5  # 중립적인 확률 반환
    name_vectorized = loaded_vectorizer.transform([name])
    probability = loaded_model.predict_proba(name_vectorized)[0][1]
    return probability

# 3. 이름만 추출하고 gender 특성 추가
train_df['FirstName'] = train_df['Name'].apply(extract_first_name)
test_df['FirstName'] = test_df['Name'].apply(extract_first_name)

train_df['GenderProbability'] = train_df['FirstName'].apply(predict_gender_probability)
test_df['GenderProbability'] = test_df['FirstName'].apply(predict_gender_probability)

# 4. Age 결측값을 평균값으로 대체
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)

# 5. 기존 전처리 진행: train_df
train_df[['Deck', 'CabinNum', 'Side']] = train_df['Cabin'].str.split('/', expand=True)
train_df.drop(columns=['Name', 'FirstName'], inplace=True)
train_df = pd.get_dummies(train_df, columns=['HomePlanet', 'Destination', 'Deck', 'Side', 'VIP', 'CryoSleep'], drop_first=True)

# GenderProbability 추가
X = train_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GenderProbability'] + list(train_df.columns[-10:])]
y = train_df['Transported'].astype(int)

# 6. 기존 전처리 진행: test_df
test_df[['Deck', 'CabinNum', 'Side']] = test_df['Cabin'].str.split('/', expand=True)
test_df.drop(columns=['Name', 'FirstName'], inplace=True)
test_df = pd.get_dummies(test_df, columns=['HomePlanet', 'Destination', 'Deck', 'Side', 'VIP', 'CryoSleep'], drop_first=True)

# 학습과 테스트에 필요한 열을 맞추기 위해 train_df에 없는 열을 test_df에 추가
for col in X.columns:
    if col not in test_df:
        test_df[col] = 0

# 열 순서 맞추기
test_df = test_df[X.columns]

# 7. LightGBM 모델 학습
model = LGBMClassifier(
    n_estimators=1768,
    max_depth=13,
    learning_rate=0.006759066453943107,
    num_leaves=36,
    subsample=0.8588280473652004,
    colsample_bytree=0.6947951715596529,
    random_state=42
)
model.fit(X, y)  # 전체 데이터를 사용해 학습

# 8. 테스트 데이터에 대한 예측
test_predictions = model.predict(test_df)
test_predictions_bool = test_predictions.astype(bool)  # Boolean 값으로 변환

# 9. 결과 파일 생성
results_df = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Transported": test_predictions_bool
})
results_df.to_csv("submission.csv", index=False)

print("테스트 데이터에 대한 결과 파일 'submission.csv'가 생성되었습니다.")
