# Google Play Store Sentiment Analysis
This code fetches reviews from the Google Play Store API, performs pre-processing and sentiment analysis on the reviews, and classifies them into categories based on the topics of app interface and load time. It also generates word clouds and plots graphs of positive and negative sentiments for each category and a summary overlapping graph.


## Dependencies
This code requires the following dependencies:
pandas
nltk
wordcloud
matplotlib
seaborn


## Explanation
The code can be broken down into several sections:

### Data Retrieval
The code first retrieves the reviews from the Google Play Store API using the provided API key and app ID. It then extracts the text and date fields from the reviews and converts them into a Pandas DataFrame.

### Data Pre-Processing
The text data is pre-processed by removing stop words and performing tokenization using the NLTK library. The sentiment of each review is analyzed using the SentimentIntensityAnalyzer class from the NLTK library.

### Classification
The code classifies the sentiment analysis into categories based on the topics of app interface and load time. The topics are defined as a dictionary of keywords for each category.

### Visualization
The code generates word clouds and plots graphs of positive and negative sentiments for each category, as well as a summary overlapping graph. The visualization is done using the WordCloud, Matplotlib, and Seaborn libraries.

## Future Work
Support for more topics and keywords.
Use machine learning algorithms to improve classification accuracy.
Implement a web interface to display the visualizations.
