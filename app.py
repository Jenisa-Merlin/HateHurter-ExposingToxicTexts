from flask import Flask, request, jsonify, render_template
import joblib  # You may need to install joblib if you haven't already

app = Flask(__name__)
# Load your CRF model here
# Replace this with code to load your CRF model

# Example of loading a model saved with joblib
# Load your CRF model from crf_model.pkl
crf_model = joblib.load("crf_model.pkl")


@app.route('/')
def front():
    return render_template('front.html')

@app.route('/api/detect-hate-span', methods=['POST'])
def detect_hate_span():
    # Get the input sentence from the request
    data = request.get_json()
    sentence = data.get('sentence', '')

    # Use your CRF model to predict the hate span
    predicted_tags = predict_hate_span(sentence)

    # Return the results as JSON
    result = {
        "sentence": sentence,
        "hateSpeechWords": predicted_tags
    }

    return jsonify(result)
# Define the word2features function
def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features


# Define the sent2features function in your Flask application
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def predict_hate_span(sentence):
    # Tokenize the sentence (modify this based on your tokenization method)
    sentence_tokens = sentence.split()
    
    # Extract features from the sentence using the loaded sent2features function
    sentence_features = [sent2features(sentence_tokens)]
    
    # Use the CRF model to predict NER tags
    predicted_tags = crf_model.predict(sentence_features)[0]
    
    return predicted_tags


if __name__ == '__main__':
    app.run(port=5500)
