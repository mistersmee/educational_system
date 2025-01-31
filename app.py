# app.py
from flask import Flask, render_template, request, jsonify
from educational_system import ContentConfig, ContentPipeline
import logging

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

        config = ContentConfig(
            subject=subject,
            grade_level=grade_level
        )
        
        pipeline = ContentPipeline(config)
        result = pipeline.process_content(prompt)

        return jsonify({
            'status': 'success',
            'data': {
                'original_content': result['generated_content'],
                'refined_content': result['refined_content']['refined'],
                'bias_report': result['bias_report'],
                'metadata': result['metadata']
            }
        })
    except Exception as e:
        logger.error(f"Error processing content: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
