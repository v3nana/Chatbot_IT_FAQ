Step-by-Step Guide to Run Your Course FAQ Chatbot

1. Install Required Software
- Download and Install:
- Visual Studio Code
- Python 3.10.11 (64-bit)

Ensure Python is added to PATH during installation.
- Verify installation by running:
  python --version
  pip --version

2. Create a Virtual Environment
- Navigate to your project folder:
  cd C:\Users\Joana\Downloads\chatbot_caballeReyes ##change this with your project folder

Create the virtual environment:
- python -m venv .venv

Activate the virtual environment:
- Windows (Command Prompt):
  .venv\Scripts\activate

- Windows (PowerShell):
  .\.venv\Scripts\Activate.ps1

- Mac/Linux:
  source .venv/bin/activate

3. Install Required Dependencies
- Run the following command to install the required Python packages:
  pip install tensorflow numpy nltk flask scikit-learn

If you have a requirements.txt file:
- pip install -r requirements.txt

4. Train the Chatbot
- Run the training script:
  python train.py

This will:
- Process data.json
- Train a machine learning model
- Save files: chatbot_model.h5, words.pkl, and classes.pkl

6. Run the Chatbot Server
- Start the chatbot application:
  python app.py

If successful, you should see:
- Running on http://127.0.0.1:5000

6. Access the Chatbot
- Open your web browser and visit:
  http://127.0.0.1:5000

This will display the chatbot interface.

8. Chatbot Frontend (index.html)
- Your index.html file will display:
  User messages
  Bot responses
  Send button

If index.html is not displaying, ensure it is inside the project folder and app.py is correctly serving it.

8. Troubleshooting Errors
- "ModuleNotFoundError" (e.g., NLTK missing):
  pip install nltk

If other modules are missing:
- pip install tensorflow flask numpy scikit-learn

Virtual Environment Not Found:
- python -m venv .venv

- Then activate it before running commands.
  
(IF) Model Not Found (chatbot_model.h5 missing):
- Make sure train.py runs successfully before app.py
