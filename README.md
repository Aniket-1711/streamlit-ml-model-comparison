# 🍔 Zomato Restaurant Rating Predictor 🍔

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/) [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/stable/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive web application built with Streamlit to predict the rating category of restaurants in Bangalore. This dashboard compares four different machine learning models to determine the most effective one for the task.

---

### 🚀 Live Demo

**[➡️ View the live application here!](https://aniket-1711-streamlit-ml-model-comparison-main-qmunx8.streamlit.app/)**

---

### ✨ Features

* **Interactive UI**: A clean, user-friendly interface built with Streamlit.
* **Data Cleaning**: The app performs a full, automated data cleaning and preprocessing pipeline on the Zomato dataset.
* **Individual Model Analysis**: Explore the performance of four different models (Logistic Regression, KNN, SVM, Naive Bayes) with detailed Classification Reports and Confusion Matrices.
* **Model Comparison**: A dedicated page to visually compare the accuracy of all four models.
* **Final Recommendation**: The app automatically identifies and recommends the best-performing model for the given dataset.

### 🛠️ Tech Stack

* **Language**: Python
* **Web Framework**: Streamlit
* **Data Manipulation**: Pandas, NumPy
* **Data Visualization**: Seaborn, Matplotlib
* **Machine Learning**: Scikit-learn

### 🔧 Setup and Installation

Follow these steps to run the project locally on your machine.

**1. Clone the Repository**

```bash
git clone [https://github.com/Aniket-1711/streamlit-ml-model-comparison.git](https://github.com/Aniket-1711/streamlit-ml-model-comparison.git)
cd streamlit-ml-model-comparison


2. Set up Git LFS

This project uses Git LFS to handle the large dataset.
# Install Git LFS (if you haven't already)
git lfs install

# Pull the large files from LFS storage
git lfs pull