# TASK 2 : CREDIT CARD FRAUD DETECTION
# CodSoft Machine Learning Internship


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



# Load Dataset

data = pd.read_csv("fraudTrain.csv")


print("Dataset Preview:")

print(data.head())



# Select useful columns only

data = data[
    [
        "amt",
        "lat",
        "long",
        "city_pop",
        "unix_time",
        "merch_lat",
        "merch_long",
        "is_fraud"
    ]
]



# Split input and output

X = data.drop(
    "is_fraud",
    axis=1
)


y = data["is_fraud"]



# Split training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Create model

model = LogisticRegression(
    max_iter=1000
)



# Train model

model.fit(
    X_train,
    y_train
)



# Predict

prediction = model.predict(
    X_test
)



# Accuracy

print("\nAccuracy:")

print(
    accuracy_score(
        y_test,
        prediction
    )
)



print("\nClassification Report:")

print(
    classification_report(
        y_test,
        prediction
    )
)



# Test sample transaction

sample = X_test.iloc[0].values.reshape(1,-1)

result = model.predict(sample)


if result[0] == 1:
    print("\nPrediction: FRAUD TRANSACTION 🚨")

else:
    print("\nPrediction: LEGITIMATE TRANSACTION ✅")