# streamlit-ml-model-comparison
🍔 Zomato Restaurant Rating Predictor 🍔
An interactive web application built with Streamlit to predict the rating category of restaurants in Bangalore. This dashboard compares four different machine learning models to determine the most effective one for the task.

🚀 Live Demo
https://aniket-1711-streamlit-ml-model-comparison-main-qmumv8.streamlit.app/

✨ Features
Interactive UI: A clean, user-friendly interface built with Streamlit.

Data Cleaning: The app performs a full, automated data cleaning and preprocessing pipeline on the Zomato dataset.

Individual Model Analysis: Explore the performance of four different models (Logistic Regression, KNN, SVM, Naive Bayes) with detailed Classification Reports and Confusion Matrices.

Model Comparison: A dedicated page to visually compare the accuracy of all four models.

Final Recommendation: The app automatically identifies and recommends the best-performing model for the given dataset.

🛠️ Tech Stack
Language: Python

Web Framework: Streamlit

Data Manipulation: Pandas, NumPy

Data Visualization: Seaborn, Matplotlib

Machine Learning: Scikit-learn

🔧 Setup and Installation
Follow these steps to run the project locally on your machine.

1. Clone the Repository

git clone [https://github.com/Aniket-1711/streamlit-ml-model-comparison.git](https://github.com/Aniket-1711/streamlit-ml-model-comparison.git)
cd streamlit-ml-model-comparison

2. Set up Git LFS

This project uses Git LFS to handle the large dataset.

# Install Git LFS (if you haven't already)
git lfs install

# Pull the large files from LFS storage
git lfs pull

3. Install Dependencies

It's recommended to use a virtual environment.

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required libraries
pip install -r requirements.txt

4. Run the Streamlit App

streamlit run main.py

The application should now be open and running in your web browser!

📂 Project Structure
├── .gitattributes        # Configures Git LFS
├── CODE_EXPLANATION.md   # Detailed explanation of the code
├── LICENSE               # MIT License
├── main.py               # The main Streamlit application script
├── README.md             # This file
├── requirements.txt      # List of Python dependencies
└── zomato.csv            # The dataset file (tracked by Git LFS)

This project was built as a hands-on learning experience to master data cleaning, machine learning model comparison, and interactive dashboard creation with Streamlit.