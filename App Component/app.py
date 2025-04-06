from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load environment variables
load_dotenv()

# Define the skin condition classes
SKIN_CONDITIONS = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Dermatofibroma',
    'Melanoma',
    'Nevus',
    'Pigmented Benign Keratosis',
    'Seborrheic Keratosis',
    'Squamous Cell Carcinoma',
    'Vascular Lesion'
]

print("Starting application initialization...")
start_time = time.time()

# Initialize the ImageDataGenerator with the same parameters as training
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.5,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255
)

# Load the pre-trained model
print("\nLoading model...")
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.h5')
model = load_model(model_path)
print("Model loaded successfully.")

# Initialize Flask app
app = Flask(__name__)

# Initialize language model only when needed
text_generator = None

def load_language_model():
    global text_generator
    if text_generator is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            language_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            text_generator = pipeline("text-generation", model=language_model, tokenizer=tokenizer)
        except Exception as e:
            print(f"Warning: Could not load language model. Using basic responses instead. Error: {str(e)}")
            text_generator = None

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    print(f"Preprocessed image shape: {img_array.shape}")

    
    # Calculate mean and std for standardization
    img_mean = img_array.mean()
    img_std = img_array.std()
    
    # Standardize the image
    img_array = (img_array - img_mean) / img_std
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_condition_info(condition):
    """Get detailed information about each skin condition"""
    info = {
        'Actinic Keratosis': {
            'severity': 'Moderate',
            'description': 'A rough, scaly patch on the skin caused by years of sun exposure.',
            'recommendations': [
                'Protect your skin from further sun damage',
                'Use prescribed topical medications',
                'Consider cryotherapy or other treatments',
                'Regular dermatologist check-ups'
            ]
        },
        'Basal Cell Carcinoma': {
            'severity': 'Serious',
            'description': 'The most common type of skin cancer, usually appears as a flesh-colored, pearl-like bump.',
            'recommendations': [
                'Immediate consultation with a dermatologist',
                'Surgical removal may be necessary',
                'Regular skin cancer screenings',
                'Strict sun protection measures'
            ]
        },
        'Dermatofibroma': {
            'severity': 'Mild',
            'description': 'A common, harmless growth that usually appears as a hard, raised bump.',
            'recommendations': [
                'No treatment necessary unless bothersome',
                'Monitor for changes in size or color',
                'Protect from trauma',
                'Surgical removal if desired'
            ]
        },
        'Melanoma': {
            'severity': 'Critical',
            'description': 'The most dangerous form of skin cancer, can spread to other parts of the body.',
            'recommendations': [
                'Immediate medical attention required',
                'Surgical removal is typically necessary',
                'Regular full-body skin examinations',
                'Follow-up care and monitoring'
            ]
        },
        'Nevus': {
            'severity': 'Mild',
            'description': 'A common mole, usually harmless but should be monitored for changes.',
            'recommendations': [
                'Regular self-examination',
                'Document any changes',
                'Protect from sun exposure',
                'Annual skin checks'
            ]
        },
        'Pigmented Benign Keratosis': {
            'severity': 'Mild',
            'description': 'A harmless growth that appears as a waxy, scaly, slightly raised growth.',
            'recommendations': [
                'No treatment necessary unless desired',
                'Monitor for changes',
                'Protect from sun damage',
                'Cosmetic removal if desired'
            ]
        },
        'Seborrheic Keratosis': {
            'severity': 'Mild',
            'description': 'A common, harmless skin growth that can appear waxy and scaly.',
            'recommendations': [
                'No treatment necessary unless bothersome',
                'Avoid irritation',
                'Cosmetic removal options available',
                'Monitor for changes'
            ]
        },
        'Squamous Cell Carcinoma': {
            'severity': 'Serious',
            'description': 'A common form of skin cancer that develops in the squamous cells.',
            'recommendations': [
                'Prompt medical evaluation',
                'Surgical removal typically needed',
                'Regular follow-up examinations',
                'Enhanced sun protection'
            ]
        },
        'Vascular Lesion': {
            'severity': 'Moderate',
            'description': 'Abnormal cluster of blood vessels that can appear as red marks on the skin.',
            'recommendations': [
                'Evaluation by a dermatologist',
                'Consider laser treatment options',
                'Protect from sun exposure',
                'Monitor for changes'
            ]
        }
    }
    return info.get(condition, {})

def get_analysis_text(condition, confidence):
    """Generate analysis text based on the classification result"""
    condition_info = get_condition_info(condition)
    
    if text_generator:
        try:
            prompt = f"""
            Based on the skin analysis:
            - Condition: {condition}
            - Confidence: {confidence:.2f}%
            - Severity: {condition_info['severity']}
            
            Medical analysis:"""
            
            response = text_generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
            return response, condition_info['recommendations']
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return get_fallback_analysis(condition, confidence)
    else:
        return get_fallback_analysis(condition, confidence)

def get_fallback_analysis(condition, confidence):
    """Provide a basic analysis when the language model is not available"""
    condition_info = get_condition_info(condition)
    
    analysis = f"""The analysis indicates signs of {condition} with {confidence:.1f}% confidence. 
    
    {condition_info['description']}
    
    Severity Level: {condition_info['severity']}
    
    Please consult a healthcare professional for a proper medical evaluation."""
    
    return analysis, condition_info['recommendations']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('result.html',
                             result="Error",
                             detailed_analysis="No file was uploaded.",
                             recommendations="Please try again with an image file.")
    # Filter top predictions with confidence above a threshold


    file = request.files['file']
    if not file:
        return render_template('result.html',
                             result="Error",
                             detailed_analysis="No file was selected.",
                             recommendations="Please try again with an image file.")
    
    print(f"File received: {file.filename}")

    try:
        # Process image and get predictions
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        preprocessed = preprocess_image(file)
        predictions = model.predict(preprocessed, verbose=0)
        print(f"Raw predictions: {predictions}")

        
        # Get top 3 predictions
        # Filter top predictions with confidence above a threshold
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_probs = predictions[0][top_3_indices]

        # Apply a threshold for displaying predictions (e.g., ignore < 5% probability)
        top_3_indices = [idx for idx, prob in zip(top_3_indices, top_3_probs) if prob >= 0.05]
        top_3_probs = [predictions[0][idx] for idx in top_3_indices]

        
        # Get the top prediction
        class_idx = top_3_indices[0]
        confidence = top_3_probs[0] * 100
        condition = SKIN_CONDITIONS[class_idx]
        
        # Generate analysis
        detailed_analysis, recommendations = get_fallback_analysis(condition, confidence)
        
        # Add top 3 predictions to analysis
        top_3_analysis = "\n\nTop 3 possibilities:\n"
        for idx, prob in zip(top_3_indices, top_3_probs):
            top_3_analysis += f"- {SKIN_CONDITIONS[idx]}: {prob*100:.2f}%\n"
        detailed_analysis = detailed_analysis + top_3_analysis
        
        return render_template('result.html',
                             result=condition,
                             confidence=confidence,
                             detailed_analysis=detailed_analysis,
                             recommendations=recommendations)

    except Exception as e:
        error_message = "The image could not be processed correctly. Please try with a different image."
        return render_template('result.html',
                             result="Error",
                             detailed_analysis=error_message,
                             recommendations="Please try again with a different image.")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # Save the file temporarily
        temp_path = "temp_upload.jpg"
        file.save(temp_path)
        
        try:
            # Preprocess the image
            processed_image = preprocess_image(temp_path)
            
            # Get model predictions
            predictions = model.predict(processed_image)
            print(f"\nRaw prediction scores: {predictions[0]}")
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_probabilities = predictions[0][top_3_indices]
            
            results = []
            for idx, prob in zip(top_3_indices, top_3_probabilities):
                results.append({
                    'condition': SKIN_CONDITIONS[idx],
                    'probability': float(prob),
                    'percentage': f"{prob * 100:.2f}%"
                })
            
            print("\nTop 3 predictions:")
            for result in results:
                print(f"{result['condition']}: {result['percentage']}")
            
            return render_template('result.html', 
                                 results=results,
                                 image_path=temp_path)
                                 
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'})
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)
