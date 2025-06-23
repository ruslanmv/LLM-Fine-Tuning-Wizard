# app.py
from flask import Flask, render_template, request, jsonify
import os
import threading
import json

app = Flask(__name__)

# --- Configuration ---
# It's good practice to have configurations in one place.
# For a real application, consider using a separate config file.
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Routes ---

@app.route('/')
def index():
    """
    Renders the main page of the wizard.
    """
    return render_template('index.html')

@app.route('/start-finetuning', methods=['POST'])
def start_finetuning_route():
    """
    This endpoint will kick off the fine-tuning process.
    For now, it just simulates the start of the process.
    """
    data = request.json
    print("Received data:", data) # For debugging

    # In a real app, you would start the finetuning script here.
    # We will implement the actual call to finetuning.py later.

    # Example of how you might call the finetuning script in a background thread
    # from finetuning import run_finetuning
    # finetuning_thread = threading.Thread(target=run_finetuning, args=(data,))
    # finetuning_thread.start()

    return jsonify({"status": "success", "message": "Fine-tuning process started!"})


if __name__ == '__main__':
    # Running the app in debug mode is helpful for development.
    # In a production environment, you would use a proper WSGI server like Gunicorn.
    app.run(debug=True, port=5001)