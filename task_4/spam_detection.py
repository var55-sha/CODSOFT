# TASK 4 : SPAM SMS DETECTION
# CodSoft Machine Learning Internship

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Load Dataset

data = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)


# Remove unwanted columns

data = data.iloc[:,0:2]


# Rename columns

data.columns = [
    "label",
    "message"
]


print("Dataset Preview")
print(data.head())


# Convert labels

data["label"] = data["label"].map(
    {
        "ham":0,
        "spam":1
    }
)


# Input and Output

X = data["message"]

y = data["label"]


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Convert text into numbers

tfidf = TfidfVectorizer(
    stop_words="english"
)

X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)


# Create ML Model

model = MultinomialNB()


# Train model

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


print("\nReport:")
print(
    classification_report(
        y_test,
        prediction
    )
)


# Test Own SMS

sms = [
    "Congratulations! You won free money click now"
]

sms = tfidf.transform(sms)


result = model.predict(sms)


if result[0] == 1:
    print("\nPrediction: SPAM SMS")

else:
    print("\nPrediction: NORMAL SMS")