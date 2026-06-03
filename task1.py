
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("titanic.csv")
print("dataset:",data)

#dropping unnecessary columns

data=data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
print("after dropping columns:",data)

#targetted columns

features= ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

X = data[features]
y = data['Survived']

#splitting data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Columns
categorical_cols = ['Sex', 'Embarked']
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Numerical preprocessing data
numerical_transformer = SimpleImputer(strategy='median')

# Categorical preprocessing data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Modeling
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train model
clf.fit(X_train, y_train)

# Predictions from dataset
y_pred = clf.predict(X_test)

# Accuracy checking
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# New passenger details to check whether the model is working or not
new_passenger = pd.DataFrame({
    'Pclass': [1],
    'Sex': ['female'],
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [100],
    'Embarked': ['S']
})

# Predict survival
prediction = clf.predict(new_passenger)

# Output result

if prediction[0] == 1:
    print("The passenger would survive.")
else:
    print("The passenger would not survive.")

print(classification_report(y_test, y_pred))
