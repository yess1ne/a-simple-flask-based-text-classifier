from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample data
texts = ["I love this movie", "I hate this thing", "Amazing story", "This is bad", "Awesome experience", "Worst movie ever"]
labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

# Vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)#ML models don't understand words, but numbers, for each input text, there is an 
#associated vector that counts how often each word appears,Scans your dataset for all unique words → builds a vocabulary.
#pretreated data,This is called a Bag-of-Words model, example : texts = ["I love this movie", "I hate this thing"]
#vocabulary : ["hate", "love", "movie", "thing"], "I love this movie" becomes: [0, 1, 1, 0]

# Model
model = MultinomialNB()
model.fit(X, labels)
#MultinomialNB = Multinomial Naive Bayes — a classic and fast text classification algorithm.
# Intuition:
#It assumes that features (words) are independent — a simplification that works surprisingly well in practice.
#Multinomial version is best when features are counts (like word frequency).
# It’s a good default model for text classification.
# Save model & vectorizer
joblib.dump(model, "model/classifier.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model saved!")
