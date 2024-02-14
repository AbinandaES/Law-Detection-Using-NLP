import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import nltk

df=pd.read_csv(r"C:\Users\sharo\Downloads\IPC - Sheet1 (1).csv")

# Load BERT tokenizer and model for sequence classification
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['section_title'], df['Section'], test_size=0.2, random_state=42)

# Define a custom transformer to tokenize using BERT
class BertTokenizerTransformer:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        tokenized_inputs = bert_tokenizer(X, max_length=37, pad_to_max_length=True)
        return tokenized_inputs['input_ids']
#Build a pipeline with BERT tokenization and a Random Forest Classifier
model_pipeline = Pipeline([
    ('tokenizer', BertTokenizerTransformer()),
    ('classifier', RandomForestClassifier())
])

# Train the model
X_train_tokenized = model_pipeline.named_steps['tokenizer'].transform(list(X_train.values))
model_pipeline.named_steps['classifier'].fit(X_train_tokenized, y_train)


from nltk.tokenize import word_tokenize
nltk.download('punkt')

while True:
    input_crime = input("Enter the description of the crime: ")

    if input_crime.lower() == 'exit':
        break

    # Tokenize the input sentence
    words = word_tokenize(input_crime)

    # Extract the words that are present in the section titles
    valid_words = [word.lower() for word in words if word.lower() in df['section_title'].str.lower().values]

    # Join the remaining words into a sentence
    cleaned_input = ' '.join(valid_words)

    # Tokenize the cleaned input using BERT tokenizer
    input_ids = model_pipeline.named_steps['tokenizer'].transform([cleaned_input])

    # Make predictions using the trained model
    predicted_ipc_section = model_pipeline.named_steps['classifier'].predict(input_ids)[0]

    # Find the corresponding punishment for the predicted IPC section
    corresponding_punishment = df.loc[df['Section'] == predicted_ipc_section, 'Punishment '].values[0]

    # Display the predicted IPC section and its corresponding punishment
    print(f'\nFor crime: "{input_crime}"')
    print(f'Predicted IPC Section: {predicted_ipc_section}')
    print(f'Corresponding Punishment: {corresponding_punishment}')

import joblib

# Assuming 'model' is your trained machine learning model
# Save the model to a file
joblib.dump(model_pipeline.named_steps['classifier'], 'IPC Law Detection.joblib')