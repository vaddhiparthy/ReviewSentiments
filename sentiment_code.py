import requests
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# App and API details
app_id = "com.example.gamingapp"
api_key = "API_KEY"

# Define the topics for sentiment analysis
topics = {
    "app_interface": ["interface", "design", "graphics"],
    "load_time": ["loading", "performance", "speed"]
}

# Define the stop words
stop_words = set(stopwords.words("english"))

# Define the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Fetch the reviews from the Google Play Store API
url = f"https://android-reviews.googleapis.com/package_name/{app_id}/reviews?auth={api_key}"
response = requests.get(url).json()

# Parse the reviews and extract the text and date fields
reviews = []
for review in response["reviews"]:
    text = review["text"]
    date = review["at"]
    reviews.append({"text": text, "date": date})

# Convert the reviews to a Pandas DataFrame
data = pd.DataFrame(reviews)

# Define a function to clean the text
def clean_text(text):
    # Tokenize the text
    words = word_tokenize(text.lower())

    # Remove stop words and punctuation
    words = [word for word in words if word.isalpha() and word not in stop_words]

    # Join the remaining words back into a string
    clean_text = " ".join(words)

    return clean_text

# Clean the text in the dataset
data["clean_text"] = data["text"].apply(clean_text)

# Analyze the sentiment of the cleaned text
data["sentiment"] = data["clean_text"].apply(lambda text: sia.polarity_scores(text)["compound"])

# Classify the sentiment analysis into categories based on topics
for topic, keywords in topics.items():
    data[topic] = data["text"].apply(lambda text: any(keyword in text.lower() for keyword in keywords))

# Generate word clouds and plot graphs of positive and negative sentiments for each category
for topic in topics:
    positive_text = " ".join(data[data[topic]]["clean_text"][data["sentiment"] > 0])
    negative_text = " ".join(data[data[topic]]["clean_text"][data["sentiment"] < 0])
    
    fig, axs = plt.subplots(ncols=2, figsize=(20, 8))
    plt.suptitle(f"{topic.capitalize()} Sentiment Analysis")
    
    # Plot the word clouds
    wc = WordCloud(width=800, height=400, background_color="white", colormap="tab10").generate(positive_text)
    axs[0].imshow(wc, interpolation="bilinear")
    axs[0].set_title("Positive Sentiments")
    axs[0].axis("off")
    
    wc = WordCloud(width=800, height=400, background_color="white", colormap="tab10").generate(negative_text)
    axs[1].imshow(wc, interpolation="bilinear")
    axs[1].set_title("Negative Sentiments")
    axs[1].axis("off")
    
    # Plot the positive and negative sentiment graphs
    positive_data = data
