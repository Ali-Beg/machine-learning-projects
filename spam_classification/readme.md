# ✉️ Spam Email Classification

This project is a **Spam Email Classification System** that uses various machine learning models to determine whether an email is spam or ham (not spam). The project involves data preprocessing, model training, and evaluation to identify the best-performing model. The application has been deployed on Streamlit for an interactive user experience.

**Streamlit App:** [Spam Classification](https://my-spam-classification.streamlit.app/)


## 🧠 How It Works

### 1. **Data Collection**
The dataset used for this project contains a collection of emails labeled as either spam or ham. The data is sourced from [insert data source, if applicable], which includes features like:
- Email content (text)
- Labels indicating whether the email is spam or ham

### 2. **Data Preprocessing**
Before training the models, the data undergoes several preprocessing steps:
- **Text Cleaning:** Emails are cleaned by removing stop words, punctuation, and special characters.
- **Tokenization:** The email content is tokenized into individual words or tokens.
- **Vectorization:** The text data is transformed into numerical vectors using techniques like TF-IDF.

### 3. **Model Training**
Several machine learning models are trained on the preprocessed data to classify emails as spam or ham. The models include:
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Multinomial Naive Bayes**
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **AdaBoost**
- **Bagging**
- **Extra Trees**
- **Gradient Boosting**
- **XGBoost**

### 4. **Model Evaluation**
Each model is evaluated based on accuracy, precision, recall, and F1-score. Among all the models tested, **Multinomial Naive Bayes** performed the best, showing the highest accuracy and consistency in classification. While stacking and voting methods were explored, Multinomial Naive Bayes still provided the most promising results.

### 5. **Deployment on Streamlit**
The final model was deployed using **Streamlit**, allowing users to input email content and receive predictions on whether the email is spam or ham.

## 💻 Tech Stack

- **Python:** Core programming language used.
- **Pandas & NumPy:** Libraries for data manipulation and analysis.
- **Scikit-learn:** Used for model training, evaluation, and vectorization.
- **Streamlit:** Frontend framework for creating an interactive web app.
- **NLTK:** Natural Language Toolkit for text preprocessing.
- **XGBoost:** Advanced implementation of gradient boosting for performance evaluation.

## 🛠️ Features

- **Spam or Ham Classification:** Users can input email content to determine if it is spam or ham.
- **Model Comparison:** Results from various models are compared, with the best model highlighted.
- **Interactive UI:** User-friendly interface powered by Streamlit.

## 🚀 How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Ali-Beg/machine-learning-projects/tree/main/spam_classification
   cd spam_classification
   ```

2. **Install Dependencies:**
   Install the required Python packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   Start the Streamlit application by running:
   ```bash
   streamlit run app.py
   ```

4. **Classify an Email:**
   Use the input box in the Streamlit app to enter email content and receive a spam or ham classification.

## 🔧 Key Files

- `app.py`: The main application file for Streamlit.
- `mnb_models.pkl`: Serialized models for deployment.
- `vectorizer.pkl`:  containing text vectorizing functions.
- `README.md`: Documentation for the project.

## 📊 Example Outputs

1. **User Input: "Congratulations, you've won a $1,000 gift card!"**
   - **Prediction:** Spam

2. **User Input: "Meeting rescheduled to 10 AM tomorrow."**
   - **Prediction:** Ham

## 📈 Data Insights (EDA)

- **Common Words in Spam:** "win", "free", "prize", "congratulations"
- **Common Words in Ham:** "meeting", "schedule", "thank you", "regards"

## 🤝 Contributions

Feel free to contribute to the project by opening issues and submitting pull requests. Feedback is welcome!
