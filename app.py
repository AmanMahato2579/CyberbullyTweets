from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import emoji

# NLTK downloads (do this only once if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
with open('classifier_lr_optimized.pickle', 'rb') as f:
    model = pickle.load(f)

with open('tfidfmodelUNIGRAM.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing functions
def expand_contractions(text):
    dict_conts = {"ain't": "are not", "'s": " is", "aren't": "are not", "'re": " are", "'t": " not", " nt ": " not ", " u ":" you "}
    re_cont = re.compile('(%s)' % '|'.join(dict_conts.keys()))
    return re_cont.sub(lambda match: dict_conts[match.group(0)], text)

def expand_hashtag(text):
    pattern = re.compile(r'#(\w+)')
    return re.sub(pattern, lambda m: re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', m.group(1)), text)

def pre_process_regex(text):
    tweet_tokenizer = TweetTokenizer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(['rt'])

    mention_pattern = re.compile(r'(?:rt)?@(\w+)')
    link_pattern = re.compile(r'https?://\S+|(www\.\S+)|(bit\.ly/\S+)|tinyurl\.\S+')

    text = expand_hashtag(text)
    text = text.lower()
    text = re.sub(mention_pattern, "", text)
    text = re.sub(link_pattern, "", text)
    text = expand_contractions(text)

    text = re.sub(r':|_|-', " ", text)
    text = emoji.demojize(text)
    text = re.sub(r'[^a-zA-Z\s:]', " ", text)
    text = emoji.emojize(text)

    text = re.sub(r'\s+', " ", text)
    text = re.sub(r'\s\w\s', " ", text)
    text = re.sub(r'^\s+|\s+$', '', text)
    text = ' '.join([word for word in tweet_tokenizer.tokenize(text) if word not in stop_words])
    return text

# Prediction
def predict_tweet(tweet):
    cleaned_tweet = pre_process_regex(tweet)
    tweet_vector = vectorizer.transform([cleaned_tweet])
    prediction = model.predict(tweet_vector)[0]
    prob = model.predict_proba(tweet_vector)[0]
    classes = model.classes_
    prob_dict = {cls: round(prob[i], 3) for i, cls in enumerate(classes)}
    return prediction, prob_dict

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    prob_dict = None

    if request.method == 'POST':
        tweet = request.form['tweet']
        if tweet.strip():
            prediction, prob_dict = predict_tweet(tweet)
        else:
            prediction = "Please enter a tweet."

    return render_template('index.html', prediction=prediction, prob_dict=prob_dict)

if __name__ == '__main__':
    app.run(debug=True)
