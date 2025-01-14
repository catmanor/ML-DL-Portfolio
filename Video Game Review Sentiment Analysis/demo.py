from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from gensim.models import Phrases, LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora

# Ensure necessary NLTK data is downloaded
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load BERT model and tokenizer
model_path = './saved_model_bert'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Predict sentiment using BERT model
def predict_sentiment(text):
    if not isinstance(text, str):
        text = str(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Data processing: tokenization and n-gram creation
stop_words = set(stopwords.words('english'))

def tokenize_and_filter(review):
    # Ensure input is a string
    if not isinstance(review, str):
        review = str(review)
    tokens = word_tokenize(review)
    return [word for word in tokens if word not in stop_words and word.isalpha()]

# Upload file page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and process dataset
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        # Load and predict sentiment
        df = pd.read_csv(filepath)
        
        # Ensure 'review' column exists and convert to string
        if 'review' not in df.columns:
            return "Error: 'review' column not found in the uploaded file.", 400
        df['review'] = df['review'].astype(str)
        
        df['predicted_label'] = df['review'].apply(predict_sentiment)

        # Separate positive and negative reviews based on prediction
        positive_reviews = df[df['predicted_label'] == 1]['review'].tolist()
        negative_reviews = df[df['predicted_label'] == 0]['review'].tolist()

        # Tokenization, n-grams, and filtering for LDA
        positive_tokens = [tokenize_and_filter(review) for review in positive_reviews]
        negative_tokens = [tokenize_and_filter(review) for review in negative_reviews]

        # Apply n-gram modeling
        bigram = Phrases(positive_tokens + negative_tokens, min_count=2, threshold=100)
        trigram = Phrases(bigram[positive_tokens + negative_tokens], threshold=50)
        
        positive_reviews_ngrams = [trigram[bigram[review]] for review in positive_tokens]
        negative_reviews_ngrams = [trigram[bigram[review]] for review in negative_tokens]

        # TF-IDF filtering
        positive_reviews_text = [" ".join(review) for review in positive_reviews_ngrams]
        negative_reviews_text = [" ".join(review) for review in negative_reviews_ngrams]

        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=3, max_features=1000, stop_words='english')
        positive_tfidf = tfidf_vectorizer.fit_transform(positive_reviews_text)
        negative_tfidf = tfidf_vectorizer.transform(negative_reviews_text)

        # Filter low-frequency words
        positive_filtered_words = tfidf_vectorizer.get_feature_names_out()
        negative_filtered_words = tfidf_vectorizer.get_feature_names_out()

        positive_reviews_filtered = [[word for word in review.split() if word in positive_filtered_words] for review in positive_reviews_text]
        negative_reviews_filtered = [[word for word in review.split() if word in negative_filtered_words] for review in negative_reviews_text]

        # Remove meaningless phrases
        meaningless_phrases = {"wan_na", "gon_na", "elden_ring", "ever_played", "right_right"}
        positive_reviews_filtered = [[word for word in review if word not in meaningless_phrases] for review in positive_reviews_filtered]
        negative_reviews_filtered = [[word for word in review if word not in meaningless_phrases] for review in negative_reviews_filtered]

        # Filter by n-gram length (bi-gram and tri-gram)
        positive_reviews_filtered = [[word for word in review if len(word.split('_')) in [2, 3]] for review in positive_reviews_filtered]
        negative_reviews_filtered = [[word for word in review if len(word.split('_')) in [2, 3]] for review in negative_reviews_filtered]

        # Bag-of-words and dictionary creation for LDA
        positive_dictionary = corpora.Dictionary(positive_reviews_filtered)
        positive_dictionary.filter_extremes(no_below=2, no_above=0.8)
        positive_corpus = [positive_dictionary.doc2bow(review) for review in positive_reviews_filtered]

        negative_dictionary = corpora.Dictionary(negative_reviews_filtered)
        negative_dictionary.filter_extremes(no_below=2, no_above=0.8)
        negative_corpus = [negative_dictionary.doc2bow(review) for review in negative_reviews_filtered]

        # Train LDA model and visualize
        num_topics = 5
        positive_lda = LdaModel(positive_corpus, num_topics=num_topics, id2word=positive_dictionary, passes=20)
        negative_lda = LdaModel(negative_corpus, num_topics=num_topics, id2word=negative_dictionary, passes=20)

        # Generate pyLDAvis visualizations and save as HTML
        pyLDAvis.save_html(gensimvis.prepare(positive_lda, positive_corpus, positive_dictionary), 'static/positive_vis.html')
        pyLDAvis.save_html(gensimvis.prepare(negative_lda, negative_corpus, negative_dictionary), 'static/negative_vis.html')

        # Get the count of positive and negative reviews
        positive_count = len(positive_reviews)
        negative_count = len(negative_reviews)
        
        # Pass the counts to the results page
        return redirect(url_for('show_results', pos_count=positive_count, neg_count=negative_count))

# Show generated visualizations with counts
@app.route('/results')
def show_results():
    pos_count = request.args.get('pos_count', 0)
    neg_count = request.args.get('neg_count', 0)
    return render_template('results.html', pos_count=pos_count, neg_count=neg_count)

if __name__ == '__main__':
    app.run(debug=True)
