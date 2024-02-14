import streamlit as st
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.pipeline import Pipeline


import joblib
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BertTokenizerTransformer:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        tokenized_inputs = bert_tokenizer(X, max_length=37, pad_to_max_length=True)
        return tokenized_inputs['input_ids']
transformer=BertTokenizerTransformer()

def load_model():
    # Load the saved model
    

    loaded_model = joblib.load('IPC Law Detection.joblib')
    return loaded_model

def load_data(file_path):
    # Load the CSV data
    df = pd.read_csv(file_path)
    return df

def clean_text(text):
    # Implement text cleaning logic (e.g., removing special characters, numbers)
    cleaned_text = text  # Placeholder, replace with actual cleaning code
    return cleaned_text

def predict_ipc_section(model, input_text, df):
    # Tokenize the input sentence
    words = word_tokenize(input_text)

    # Extract the words that are present in the section titles
    valid_words = [word.lower() for word in words if word.lower() in df['section_title'].str.lower().values]

    # Join the remaining words into a sentence
    cleaned_input = ' '.join(valid_words)

    # Tokenize the cleaned input using BERT tokenizer
    input_ids = transformer.transform([cleaned_input])

    # Make predictions using the trained model
    predicted_ipc_section = model.predict(input_ids)[0]

    return predicted_ipc_section

def main():
    st.title("Law Prediction App")

    # User input for crime description
    input_crime = st.text_area("Enter the description of the crime:", "")

    if st.button("Predict"):
        if input_crime.lower() == 'exit':
            st.warning("You entered 'exit'. Please enter a crime description.")
        else:
            # Load the model and data
            loaded_model = load_model()
            df = load_data(r"C:\Users\sharo\Downloads\IPC - Sheet1 (1).csv")  # Update with your data path

            # Clean the input text
            cleaned_input = clean_text(input_crime)

            # Predict IPC section
            predicted_ipc_section = predict_ipc_section(loaded_model, cleaned_input, df)

            # Display the predicted IPC section and its corresponding punishment
            st.write(f'\nFor crime: "{input_crime}"')
            st.write(f'Predicted IPC Section: {predicted_ipc_section}')

            # Find the corresponding punishment for the predicted IPC section
            corresponding_punishment = df.loc[df['Section'] == predicted_ipc_section, 'Punishment '].values[0]
            st.write(f'Corresponding Punishment: {corresponding_punishment}')

# Run the app
if __name__ == "__main__":
    main()
