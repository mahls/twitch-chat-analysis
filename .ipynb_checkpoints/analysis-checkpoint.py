import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("twitch_chat_log.csv")

# Clean the 'Message' column
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    return text.lower()  # Lowercase for uniformity

data['Cleaned_Message'] = data['Message'].apply(clean_text)

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']  # Compound score for overall sentiment

data['Sentiment'] = data['Cleaned_Message'].apply(get_sentiment)

# Word Frequency Analysis
vectorizer = CountVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(data['Cleaned_Message'])
word_freq = X.sum(axis=0).A1
words = vectorizer.get_feature_names_out()

word_freq_df = pd.DataFrame(list(zip(words, word_freq)), columns=['Word', 'Frequency'])
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

# Plotting top words
plt.figure(figsize=(10, 6))
plt.barh(word_freq_df['Word'].head(10), word_freq_df['Frequency'].head(10), color='skyblue')
plt.xlabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.show()

# Visualizing Sentiment Distribution
plt.figure(figsize=(10, 6))
plt.hist(data['Sentiment'], bins=30, color='lightcoral', edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Distribution of Chat Messages')
plt.show()

# Analyzing User Activity
user_activity = data['Username'].value_counts().reset_index()
user_activity.columns = ['Username', 'Message_Count']

# Display and plot top 10 users with the most messages
plt.figure(figsize=(10, 6))
plt.barh(user_activity['Username'].head(10), user_activity['Message_Count'].head(10), color='lightgreen')
plt.xlabel('Message Count')
plt.title('Top 10 Active Users')
plt.show()
