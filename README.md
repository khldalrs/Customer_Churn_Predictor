Customer Churn Prediction App

An interactive Streamlit web application that predicts customer churn using multiple machine learning models. This tool helps banks identify at-risk customers and take proactive measures to improve customer retention.

Features

User-Friendly Interface: Input customer details and receive immediate churn predictions.

Multi-Model Predictions: Leverage the power of various machine learning models for robust results.

Visual Insights: Interactive charts to visualize prediction probabilities.

Personalized Explanations: Understand the factors influencing each prediction.

Automated Outreach: Generate personalized emails to encourage customers to stay.

Machine Learning Models Used

The application employs a range of machine learning models to enhance prediction accuracy:

XGBoost Classifier (xgb_model.pkl): An optimized gradient boosting algorithm known for its performance and efficiency.

Random Forest Classifier (rf_model.pkl): An ensemble of decision trees that reduces overfitting and improves accuracy.

K-Nearest Neighbors (knn_model.pkl): A simple, effective algorithm that classifies data based on proximity to neighbors.

Additional models included:

Naive Bayes Classifier (nb_model.pkl)

Decision Tree Classifier (dt_model.pkl)

Support Vector Machine (svm_model.pkl)

Voting Classifier (voting_clf.pkl): Combines multiple models for consensus prediction.

XGBoost with SMOTE (xgboost-SMOTE.pkl): Addresses class imbalance using Synthetic Minority Over-sampling Technique.

XGBoost with Feature Engineering (xgboost-featureEngineered.pkl): Enhances model performance through engineered features.

Installation

Clone the Repository

git clone https://github.com/your_username/your_repository.git
cd your_repository

Create a Virtual Environment

python -m venv venv

Activate the Virtual Environment

Windows:

venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install Dependencies

pip install -r requirements.txt

Set Up Environment Variables

Obtain your API key for the OpenAI or Groq API.

Set the environment variable:

Windows:

set GROQ_API_KEY=your_api_key_here

macOS/Linux:

export GROQ_API_KEY='your_api_key_here'

Run the Application

streamlit run main.py

Usage

Navigate to http://localhost:8501 in your web browser.

Select a customer from the dropdown or input custom details.

View churn predictions from multiple models.

Read the explanation of predictions.

Generate personalized emails for at-risk customers.

Project Structure

main.py: The main Streamlit application.

utils.py: Contains utility functions for generating charts and processing data.

Models: Pre-trained machine learning models stored as .pkl files.

churn.csv: Dataset containing customer information.

Technologies Used

Python

Streamlit

Pandas & NumPy

Scikit-learn

XGBoost

Plotly

OpenAI/Groq API
Screenshots:
![image](https://github.com/user-attachments/assets/40ae1f5c-a092-4686-b33b-90e38b64ba4e)
![image](https://github.com/user-attachments/assets/30567da4-5940-4e28-96c6-be342978d398)
![image](https://github.com/user-attachments/assets/c6c43c8a-1537-448d-9bee-d9233932b8a1)
![image](https://github.com/user-attachments/assets/39e6bade-6c75-4dc1-b150-2fb96b0a972a)



Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.
