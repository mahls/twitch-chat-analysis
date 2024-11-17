import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import datetime

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

# Sentiment Over Time
if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour

    sentiment_by_hour = data.groupby('Hour')['Sentiment'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_by_hour.index, sentiment_by_hour.values, marker='o', color='purple')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Sentiment')
    plt.title('Average Sentiment Over Time')
    plt.xticks(np.arange(0, 24, step=1))
    plt.grid(True)
    plt.show()

# Message Length Distribution
data['Message_Length'] = data['Cleaned_Message'].apply(len)

plt.figure(figsize=(10, 6))
plt.hist(data['Message_Length'], bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.title('Message Length Distribution')
plt.show()

# Sentiment by User
sentiment_by_user = data.groupby('Username')['Sentiment'].mean().reset_index()

# Plotting Sentiment by User
plt.figure(figsize=(10, 6))
plt.barh(sentiment_by_user.sort_values('Sentiment', ascending=False)['Username'].head(10),
         sentiment_by_user.sort_values('Sentiment', ascending=False)['Sentiment'].head(10), color='orange')
plt.xlabel('Average Sentiment')
plt.title('Top 10 Users by Sentiment')
plt.show()

# Top N-Grams (Bigrams)
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=10)
X_bigrams = bigram_vectorizer.fit_transform(data['Cleaned_Message'])
bigram_freq = X_bigrams.sum(axis=0).A1
bigrams = bigram_vectorizer.get_feature_names_out()

bigram_freq_df = pd.DataFrame(list(zip(bigrams, bigram_freq)), columns=['Bigram', 'Frequency'])
bigram_freq_df = bigram_freq_df.sort_values(by='Frequency', ascending=False)

# Plotting top bigrams
plt.figure(figsize=(10, 6))
plt.barh(bigram_freq_df['Bigram'].head(10), bigram_freq_df['Frequency'].head(10), color='lightcoral')
plt.xlabel('Frequency')
plt.title('Top 10 Most Frequent Bigrams')
plt.show()

# Time of Day Analysis (if 'Timestamp' is available)
if 'Timestamp' in data.columns:
    data['Hour'] = data['Timestamp'].dt.hour
    message_count_by_hour = data.groupby('Hour').size()

    plt.figure(figsize=(10, 6))
    plt.plot(message_count_by_hour.index, message_count_by_hour.values, marker='o', color='teal')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Messages')
    plt.title('User Activity Over the Day')
    plt.xticks(np.arange(0, 24, step=1))
    plt.grid(True)
    plt.show()

