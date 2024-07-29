from flask import Flask, request, jsonify
import openai
from textblob import TextBlob

# Initialize the Flask application
app = Flask(__name__)

# Configure the OpenAI API key. Replace 'your_openai_api_key' with your actual key.
openai.api_key = 'your_openai_api_key'

def analyze_sentiment(text):
    """
    Analyze the sentiment of the text using TextBlob.
    
    Args:
        text (str): The text for which to analyze sentiment.
    
    Returns:
        float: Sentiment value of the text, where -1 is negative and 1 is positive.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Returns the polarity of the sentiment

@app.route('/generate-content', methods=['POST'])
def generate_content():
    """
    Generate content using OpenAI's GPT-3 model based on the provided prompt and options.
    
    Expected POST request JSON format:
        - 'prompt': The input prompt for GPT-3.
        - 'style': (Optional) Desired style for the content. E.g., 'formal', 'casual'.
        - 'tone': (Optional) Desired tone for the content. E.g., 'informative', 'persuasive'.
        - 'num_variations': (Optional) Number of variations of content to generate.
    
    Returns:
        JSON: Contains a list of generated responses with content and sentiment analysis.
    """
    # Retrieve data from the JSON request
    data = request.get_json()
    prompt = data.get('prompt', '')
    style = data.get('style', 'neutral')  # Default style is 'neutral'
    tone = data.get('tone', 'informative')  # Default tone is 'informative'
    num_variations = int(data.get('num_variations', 1))  # Default number of variations is 1
    
    # Check if a prompt was provided
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    # Adjust the prompt to include provided style and tone
    style_prompt = f"{style} style"  # Simplified example of style
    tone_prompt = f"{tone} tone"  # Simplified example of tone
    full_prompt = f"Generate content with {style_prompt} and {tone_prompt} based on: {prompt}"

    responses = []
    # Generate multiple variations of the content
    for _ in range(num_variations):
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use the GPT-3 model 'text-davinci-003'
            prompt=full_prompt,
            max_tokens=150  # Limit the response to 150 tokens
        )
        # Get the generated text and trim whitespace
        content = response.choices[0].text.strip()
        # Analyze the sentiment of the generated content
        sentiment = analyze_sentiment(content)
        # Append the content and its sentiment to the list of responses
        responses.append({'content': content, 'sentiment': sentiment})

    # Return the list of generated responses in JSON format
    return jsonify({'responses': responses})

# Run the Flask application if the script is executed directly
if __name__ == '__main__':
    app.run(debug=True)  # Run the server in debug mode for development
