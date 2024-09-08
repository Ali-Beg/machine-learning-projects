import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix


# Load model and TF-IDF vectorizer
model = pickle.load(open('spam_classification/mnb_model.pkl', 'rb'))
tfidf = pickle.load(open('spam_classification/spam.csv', 'rb'))

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# Preprocess function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Streamlit App
st.title("📱 SMS Spam Classifier")
st.markdown("""
### Welcome to the SMS Spam Classifier app!
Enter the message below and find out if it's spam or ham (not spam). 
""")

# Text input
input_sms = st.text_area("🔤 Enter your SMS here", placeholder="Type your message...")

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.error("Please enter a valid SMS message.")
    else:
        # Preprocess the input SMS
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the SMS using the TF-IDF vectorizer
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict using the loaded model
        result = model.predict(vector_input)[0]
        
        # Display the result
        if result == 1:
            st.error("⚠️ This message is **SPAM**.")
        else:
            st.success("Message is not spam")




