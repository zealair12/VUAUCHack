# Skin Disease Classifier

A deep learning-based web application for classifying skin conditions using DenseNet201 architecture. The application can identify 9 different types of skin conditions with detailed analysis and recommendations.

## Features

- 🖼️ **Image Upload**: Drag-and-drop interface for easy image upload
- 🧠 **Deep Learning**: Uses DenseNet201 model for accurate classification
- 📊 **Detailed Analysis**: Provides top 3 predictions with confidence scores
- 💡 **Medical Insights**: Includes severity levels and recommendations for each condition
- 🎨 **Modern UI**: Clean, responsive interface built with Tailwind CSS

## Supported Conditions

The model can classify the following skin conditions:
- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented Benign Keratosis
- Seborrheic Keratosis
- Squamous Cell Carcinoma
- Vascular Lesion

## Technical Details

### Model Architecture
- Base Model: DenseNet201
- Input Shape: 128x128x3 (RGB images)
- Custom Top Layers:
  - Flatten layer
  - Dropout (0.5)
  - Dense layer (512 units, ReLU activation)
  - Output layer (9 units, softmax activation)

### Preprocessing Pipeline
- Image resizing to 128x128
- RGB conversion
- Standardization (mean subtraction and division by standard deviation)
- Batch dimension addition

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zealair12/VUAUCHack.git
cd VUAUCHack
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
cd "App Component"
pip install -r requirements.txt
```

4. Download the model:
- The model file (`model.h5`) is not included in the repository due to size limitations
- Place the model file in the root directory of the project

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000` (Will get off Local Host and Deploy to server when more data is obtained for training)

3. Upload an image of a skin condition through the web interface

4. View the analysis results, including:
   - Top prediction with confidence score
   - Top 3 possible conditions
   - Detailed medical information
   - Recommendations


## Limitations and Challenges

1. **Model Size**: The trained model file is large (261.34 MB) and cannot be hosted on GitHub directly. Users need to obtain the model file separately.

2. **Preprocessing Mismatch**: The application initially had a preprocessing mismatch between training and inference, which was fixed by implementing proper standardization.

3. **Performance Considerations**:
   - Model loading time can be significant due to the large model size
   - Inference speed depends on available hardware
   - Memory usage is high due to the DenseNet201 architecture

4. **Accuracy Limitations**:
   - The model's accuracy is limited by the training data quality and quantity
   - Performance may vary with different skin tones and lighting conditions
   - Not a replacement for professional medical diagnosis

5. **Technical Dependencies**:
   - Requires significant computational resources
   - Dependencies on specific versions of TensorFlow and other libraries
   - GPU acceleration recommended for optimal performance

## Future Improvements

1. Implement model quantization to reduce size and improve inference speed
2. Add support for batch processing of multiple images
3. Enhance preprocessing pipeline with better augmentation
4. Implement a more robust error handling system
5. Add support for different image formats and sizes
6. Develop a mobile-friendly version of the application

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DenseNet201 architecture by Huang et al.
- ISIC dataset for training data
- Flask and TensorFlow communities for their excellent tools and documentation
