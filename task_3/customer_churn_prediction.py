# TASK 3 : CUSTOMER CHURN PREDICTION
# CodSoft Machine Learning Internship


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# Load Dataset

data = pd.read_csv("Churn_Modelling.csv")


print("Dataset Preview:")
print(data.head())


# Remove unnecessary columns

data = data.drop(
    [
        "RowNumber",
        "CustomerId",
        "Surname"
    ],
    axis=1
)



# Convert text columns

encoder = LabelEncoder()


data["Geography"] = encoder.fit_transform(
    data["Geography"]
)


data["Gender"] = encoder.fit_transform(
    data["Gender"]
)



# Split input and output

X = data.drop(
    "Exited",
    axis=1
)


y = data["Exited"]



# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Create ML model

model = RandomForestClassifier()



# Train

model.fit(
    X_train,
    y_train
)



# Prediction

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



# Test customer

sample = X_test.iloc[0].values.reshape(1,-1)


result = model.predict(sample)


if result[0] == 1:

    print("\nPrediction: Customer Will Leave")

else:

    print("\nPrediction: Customer Will Stay")