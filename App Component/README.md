# Skin Disease Classifier

A web application that uses machine learning to classify skin conditions and provide detailed analysis using Hugging Face's language models.

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your Hugging Face API token:
     ```
     HUGGINGFACE_TOKEN=your_token_here
     ```
   - Never commit the `.env` file to version control

5. Run the application:
   ```bash
   python app.py
   ```

## Security Note

- The `.env` file is included in `.gitignore` to prevent accidental commits of sensitive information
- Always keep your API keys secure and never share them publicly
- For deployment, use environment variables or secure secret management systems

## Project Structure

```
App Component/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in git)
├── .gitignore         # Git ignore rules
├── templates/         # HTML templates
│   ├── index.html    # Upload page
│   └── result.html   # Results page
└── README.md         # This file
``` 