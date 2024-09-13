from flask import Flask, request, jsonify
from helper import *
import os

# load the model
classifier = load_model()

# Create Flask app
app = Flask(__name__)

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    sentiment, confidence = predict_sentiment_aspect(text, classifier)
    
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)