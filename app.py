from flask import Flask, request, jsonify
import openai
from textblob import TextBlob

app = Flask(__name__)

# Configura tu clave de API de OpenAI
openai.api_key = 'your_openai_api_key'

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # -1 (negative) to 1 (positive)

@app.route('/generate-content', methods=['POST'])
def generate_content():
    data = request.get_json()
    prompt = data.get('prompt', '')
    style = data.get('style', 'neutral')
    tone = data.get('tone', 'informative')
    num_variations = int(data.get('num_variations', 1))
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    style_prompt = f"{style} style"  # Simplified style prompt
    tone_prompt = f"{tone} tone"  # Simplified tone prompt
    
    full_prompt = f"Generate content with {style_prompt} and {tone_prompt} based on: {prompt}"

    responses = []
    for _ in range(num_variations):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=full_prompt,
            max_tokens=150
        )
        content = response.choices[0].text.strip()
        sentiment = analyze_sentiment(content)
        responses.append({'content': content, 'sentiment': sentiment})

    return jsonify({'responses': responses})

if __name__ == '__main__':
    app.run(debug=True)
