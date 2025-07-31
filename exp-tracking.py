import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Set the MLflow experiment
experiment_name = "Fake News Detection"
mlflow.set_experiment(experiment_name)

# Load the data
try:
    x_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    x_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Drop the 'Unnamed: 0' column if it exists
x_train = x_train.drop(columns=['Unnamed: 0'], errors='ignore')
y_train = y_train.drop(columns=['Unnamed: 0'], errors='ignore')
x_test = x_test.drop(columns=['Unnamed: 0'], errors='ignore')
y_test = y_test.drop(columns=['Unnamed: 0'], errors='ignore')

# Handle NaN values in the 'processed_text' column
x_train['processed_text'] = x_train['processed_text'].fillna('unknown')
x_test['processed_text'] = x_test['processed_text'].fillna('unknown')

# Ensure y_train and y_test are 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Initialize the TfidfVectorizer
tfidf_params = {
    "stop_words": "english",
    "max_df": 0.8
}
tfidf = TfidfVectorizer(**tfidf_params)

# Fit and transform the training data
x_train_tfidf = tfidf.fit_transform(x_train['processed_text'])

# Transform the test data
x_test_tfidf = tfidf.transform(x_test['processed_text'])

# Initialize the SVM Classifier model with some hyperparameters
svm_params = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": None,
    "random_state": 42
}
model = SVC(**svm_params)

# Start an MLflow run
with mlflow.start_run() as run:
    print(f"MLflow run started with run ID: {run.info.run_id}")
    
    # Train the model
    model.fit(x_train_tfidf, y_train)
    
    # Make predictions
    preds = model.predict(x_test_tfidf)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc}")
    
    # Log parameters and metrics to MLflow
    mlflow.log_param("model", "SVC")
    mlflow.log_params(svm_params)
    mlflow.log_params(tfidf_params)
    mlflow.log_metric("accuracy", acc)
    
    # Save the TfidfVectorizer
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    
    # Log the TfidfVectorizer as an artifact
    mlflow.log_artifact("tfidf.pkl")
    
    # Log the model with an input example to infer the model signature
    input_example = x_test_tfidf[0:1]  # Use the first row of the test data as an example
    mlflow.sklearn.log_model(model, "model", input_example=input_example)
    print("Model logged successfully.")