from flask import Flask, render_template, request, jsonify
from educational_system import ContentConfig, ContentPipeline
import logging
import json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        subject = data.get('subject', '')
        grade_level = int(data.get('grade_level', 5))
        prompt = data.get('prompt', '')

        # Set up the ContentConfig and run the pipeline
        config = ContentConfig(subject=subject, grade_level=grade_level)
        pipeline = ContentPipeline(config)
        pipeline.process_content(prompt)  # This writes to content_output.json

        # Read the JSON output
        with open("content_output.json", "r") as f:
            result = json.load(f)

        # Ensure clarity, coherence, and bias report are always included
        result['refined_content']['metrics'].setdefault('clarity_score', None)
        result['refined_content']['metrics'].setdefault('coherence_score', None)
        result.setdefault('bias_report', {})

        return jsonify(result)  # Send the updated JSON to the frontend

    except Exception as e:
        logger.error(f"Error processing content: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
