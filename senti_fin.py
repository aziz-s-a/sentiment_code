import streamlit as st
import pandas as pd
import nltk
import re
# Download NLTK data if not already present
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
 


st.title('Excel Sentiment Analyzer')
# Load your Excel or CSV file
uploaded_file = st.file_uploader('Upload an Excel file', type=['xlsx', 'csv'])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Function to extract messages from a specific caller
    def extract_by_caller(full_text, caller):
        if pd.isna(full_text) or pd.isna(caller):
            return ""
        messages = re.split(r"(?=\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2} - )", full_text)
        filtered_msgs = [msg.strip() for msg in messages if f"- {caller} (" in msg]
        return "\n\n".join(filtered_msgs)

    # Apply the function to your DataFrame
    df['filtered_comments'] = df.apply(lambda row: extract_by_caller(row['Additional comments'], row['Caller']), axis=1)
    # Set 'Number' column as index if it exists
    if 'Number' in df.columns:
        df.set_index('Number', inplace=True)
    # Save the filtered output in the same format as the uploaded file
    if uploaded_file.name.endswith('.csv'):
        df.to_csv("filtered_output.csv", index=True)
        filtered_path = "filtered_output.csv"
    else:
        df.to_excel("filtered_output.xlsx", index=True)
        filtered_path = "filtered_output.xlsx"

    # Use the filtered output for sentiment analysis
    st.write('Preview of filtered data:')
    st.dataframe(df.head())

    text_column = st.selectbox('Select the column containing text for sentiment analysis:', df.columns)

    # Ensure SentimentIntensityAnalyzer and vader_lexicon are defined in this scope
    sia = SentimentIntensityAnalyzer()
    vader_lexicon = sia.lexicon

    def highlight_sentiment_words(note):
        words = re.findall(r'\b\w+\b', str(note).lower())
        positive_words = []
        negative_words = []
        for word in words:
            if word in vader_lexicon:
                score = float(vader_lexicon[word])
                if score > 0:
                    positive_words.append(word)
                elif score < 0:
                    negative_words.append(word)
        return positive_words, negative_words

    def analyze_sentiment_with_explanation(note):
        scores = sia.polarity_scores(str(note))
        compound = scores['compound']
        pos = scores['pos']
        neu = scores['neu']
        neg = scores['neg']
        positive_words, negative_words = highlight_sentiment_words(note)
        if compound >= 0.05:
            sentiment = "positive"
            reason = f"High positive score ({pos:.2f}). Positive words: {', '.join(positive_words) if positive_words else 'None'}"
        elif compound <= -0.05:
            sentiment = "negative"
            reason = f"High negative score ({neg:.2f}). Negative words: {', '.join(negative_words) if negative_words else 'None'}"
        else:
            sentiment = "neutral"
            reason = f"Balanced scores: pos={pos:.2f}, neg={neg:.2f}"
        return pd.Series([sentiment, reason, pos, neu, neg, compound])

    if st.button('Analyze Sentiment'):
        df[['sentiment', 'reason', 'Positive Score', 'Neutral Score', 'Negative Score', 'Compound Score']] = \
            df[text_column].apply(analyze_sentiment_with_explanation)
        st.write('Sentiment Analysis Results:')
        # Always include Number as a column in the output if it exists (reset index if needed)
        df_reset = df.reset_index() if 'Number' in df.index.names else df
        output_columns = [col for col in ['Number', text_column, 'sentiment', 'reason', 'Positive Score', 'Neutral Score', 'Negative Score', 'Compound Score'] if col in df_reset.columns or col == text_column]
        st.dataframe(df_reset[output_columns])
        # Option to download results
        output = df_reset[output_columns]
        output.to_excel('sentiment_results.xlsx', index=False)
        with open('sentiment_results.xlsx', 'rb') as f:
            st.download_button('Download Results as Excel', f, file_name='sentiment_results.xlsx')