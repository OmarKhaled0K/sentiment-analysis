import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
import emoji
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
def load_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier

# preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters but keep emojis and Arabic characters
    text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]', ' ', text)
    return text.strip()

# decode emoji
def emoji_to_sentiment(text):
    # Expanded emoji sentiment mapping
    positive_emojis = set(['❤️', '😊', '😄', '👍', '🎉', '😍', '🥰', '😘', '🤗', '😁', '😀', '🙌', '👏', '💪', '🔥'])
    negative_emojis = set(['😢', '😠', '👎', '😞', '😡', '💔', '😭', '😒', '😓', '😔', '😖', '😫', '😩', '🙄', '😤'])

    emoji_only_text = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    if emoji_only_text:
        positive_count = sum(1 for e in emoji_only_text if e in positive_emojis)
        negative_count = sum(1 for e in emoji_only_text if e in negative_emojis)
        if positive_count > negative_count:
            return 'positive', 1.0
        elif negative_count > positive_count:
            return 'negative', 1.0
        else:
            return 'neutral', 1.0
    return None

# generate prediction
def predict_sentiment_aspect(text, classifier):
    sentences = preprocess_text(text)
    if not sentences:
        return "neutral", 0.0

    emoji_sentiment = emoji_to_sentiment(text)
    if emoji_sentiment:
        return emoji_sentiment

    try:
        results = classifier(sentences)
        logger.info(f"Raw model output for '{text}': {results}")
        
        # Analyze sentiment for each aspect (sentence)
        sentiments = []
        for result in results:
            label = result['label']
            if label == 'positive' or label == 'LABEL_2':
                sentiments.append(('positive', result['score']))
            elif label == 'negative' or label == 'LABEL_0':
                sentiments.append(('negative', result['score']))
            else:
                sentiments.append(('neutral', result['score']))
        
        # Determine overall sentiment
        positive_count = sum(1 for s, _ in sentiments if s == 'positive')
        negative_count = sum(1 for s, _ in sentiments if s == 'negative')

        if positive_count > negative_count:
            return 'positive', np.mean([score for _, score in sentiments if _ == 'positive'])
        elif negative_count > positive_count:
            return 'negative', np.mean([score for _, score in sentiments if _ == 'negative'])
        else:
            return 'neutral', np.mean([score for _, score in sentiments])

    except Exception as e:
        logger.error(f"Error processing text: {text}")
        logger.error(f"Error message: {str(e)}")
        return "neutral", 0.0
