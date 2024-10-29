
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

data = pd.read_csv('reviews.csv')
data.drop_duplicates(subset=["UserId", "ProfileName", "Time", "Text"], keep='first', inplace=True)
def tags(text):
    soup = BeautifulSoup(text, 'lxml')
    return soup.get_text()
def review(text):
    text = tags(text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\S*\d\S*", "", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text
review_corpus = [review(entry['Text']) for _, entry in tqdm(data.iterrows(), total=data.shape[0])]
data['Score'] = data['Score'].apply(lambda score: 1 if score >= 3 else 0)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
features = vectorizer.fit_transform(review_corpus).toarray()
target = data['Score'].values
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Model Performance Report:\n", classification_report(y_test, predictions))

conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
def review():
    try:
        input_review = input("Enter a review for your analysis: ").strip()
        if not input_review:
            raise ValueError("Review content cannot be empty so enter something in it")
        
        cleaned_review = review(input_review)
        transformed_review = vectorizer.transform([cleaned_review]).toarray()
        sentiment = model.predict(transformed_review)
        
        if sentiment[0] == 1:
            print("This is a Positive Review")
        else:
            print("This is a Negative Review")
    
    except Exception as error:
        print(f"An error occurred: {error}")
