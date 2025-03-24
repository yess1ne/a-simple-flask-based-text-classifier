# 🧠 A Simple Flask-Based Text Classifier
This is a demo project that uses **Flask** (Python web framework) to serve a simple **AI-powered text classifier**.  
It takes user input (e.g., a sentence or phrase) and predicts whether the sentiment is **positive** or **negative** using a **Naive Bayes** model trained on custom text data.

---
## 🚀 Features
- 📝 Submit text through a web interface
- ⚙️ Uses a trained `MultinomialNB` model from `scikit-learn`
- 🔤 Classifies text sentiment as **positive** or **negative**
- 🧪 Built with a minimal dataset but easily expandable
- 🎓 Great for learning Flask + basic ML integration
---
## 🛠️ Technologies
- [Flask](https://flask.palletsprojects.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [Joblib](https://joblib.readthedocs.io/)
---
## 📂 Project Structure
<pre><code>```bash flask-ai-app/ ├── app.py # Flask web app ├── train_model.py # Script to train and save the model ├── model/ │ ├── classifier.pkl # Trained model │ └── vectorizer.pkl # CountVectorizer instance ├── templates/ │ └── index.html # Web UI for input and prediction └── venv/ # Virtual environment (not pushed to GitHub) ```</code></pre>
  
---
## ⚙️ How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yess1ne/a-simple-flask-based-text-classifier.git
   cd a-simple-flask-based-text-classifier
   ```
2. **Create a virtual environment** *(dependence management in case you have multiple flask projects with variying versions)*:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows source venv/bin/activate # On Linux/macOS
```
3. **Install dependencies**:
```bash
pip install -r requirements.txt
```
4. **Train the model**:
```bash
python train_model.py
```
5. **Run the Flask app**:
```bash
python app.py
```
6. **Open your browser and go to http://localhost:5000** *(though you should be asble to see the link in the terminal after step 5)*
