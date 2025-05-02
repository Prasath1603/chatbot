import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load custom mental health dataset
# df = pd.read_csv("mental_health_dataset.csv")
df = pd.read_csv("mental_health_dataset.csv", quotechar='"')


# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Mental health model trained and saved successfully!")

# # train_model.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# import joblib
# import os

# # Sample dataset
# data = {
#     "text": [
#         "I am feeling very happy today!",
#         "Everything is so frustrating and annoying.",
#         "I'm just okay, nothing special.",
#         "I had a terrible day, very sad.",
#         "Life is beautiful and I'm grateful.",
#         "I'm so angry at what happened!",
#         "Today was boring and dull.",
#         "I feel relaxed and neutral.",
#         "I'm crying and feel terrible.",
#         "I'm excited for tomorrow!"
#     ],
#     "sentiment": [
#         "happy", "angry", "neutral", "sad", "happy",
#         "angry", "neutral", "neutral", "sad", "happy"
#     ]
# }

# df = pd.DataFrame(data)

# # Vectorization
# vectorizer = TfidfVectorizer(stop_words="english")
# X = vectorizer.fit_transform(df["text"])
# y = df["sentiment"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Save model and vectorizer
# os.makedirs("models", exist_ok=True)
# joblib.dump(model, "models/sentiment_model.pkl")
# joblib.dump(vectorizer, "models/vectorizer.pkl")

# print("Model trained and saved successfully!")

# # train_model.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# import joblib

# # Load dataset
# data = pd.read_csv("https://raw.githubusercontent.com/datasets/sentiment140/master/data/sentiment140.csv", encoding="latin-1", header=None)
# data.columns = ["target", "id", "date", "flag", "user", "text"]
# data = data[["text", "target"]]
# data["target"] = data["target"].map({0: "sad", 4: "happy"})

# # Preprocessing
# X = data["text"]
# y = data["target"]

# vectorizer = TfidfVectorizer(stop_words="english")
# X_vect = vectorizer.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Save model and vectorizer
# joblib.dump(model, "models/sentiment_model.pkl")
# joblib.dump(vectorizer, "models/vectorizer.pkl")