# **📝 Zomato Rating Predictor: Code Explanation**
# This document provides a detailed, step-by-step explanation of the Python code in the main.py file.

# **1. 📚 Importing Libraries**
This is the first step in any Python script. We import all the necessary "toolkits" that provide pre-built functions, so we don't have to write everything from scratch.

**streamlit: 🌐** The core framework used to build the interactive web application.

**pandas: 🐼** The primary tool for data manipulation. We use it to read the zomato.csv file into a table-like structure called a DataFrame and to perform all our data cleaning operations.

**numpy: 🔢** A fundamental library for numerical operations. Pandas is built on top of it.

**seaborn & matplotlib.pyplot: 📊** Our data visualization libraries. We use them together to create the beautiful and informative Confusion Matrix plots.

**scikit-learn (sklearn): 🧠** This is our complete machine learning toolbox.

**train_test_split: 🔪** A function to automatically divide our data into a training set and a testing set.

**LabelEncoder: 🏷️** A tool to convert text data (e.g., 'BTM Layout') into numerical data (e.g., 5).

**classification_report, confusion_matrix, accuracy_score: 📜 These are our evaluation metrics to score the models.**

**The Models: The four different  classification algorithms we are comparing( LogisticRegression, KNeighborsClassifier, SVC, GaussianNB).**

# **2. 📄 Page Configuration**
The st.set_page_config() command sets up the basic properties of our web page. This must be the first Streamlit command in the script.

**page_title:** The text that appears in the browser tab.

**page_icon:** 🍔 The small icon next to the title.

**layout="wide":** Tells the app to use the full width of the browser.

**initial_sidebar_state="expanded":** Ensures the sidebar is open by default.

# **3. 🧹 Data Loading and Preprocessing**
This is the most critical section, handled by the **load_and_preprocess_data()** function. We use the **@st.cache_data** decorator so that all these heavy operations run only once, making the app very fast. ⚡

# The Cleaning Journey:
**📥 Loading:** We start by loading the full **zomato.csv**.

**🎲 Sampling:** We take a random sample of **10,000** rows to ensure our app is fast. **random_state=42** guarantees we get the same sample every time.

**✏️ Renaming & Selection:** We rename the complex approx_cost(for two people) column to cost2plates and select only the relevant columns.

**🗑️ Handling Duplicates (Round 1):** We use **df.drop_duplicates()** to remove initial duplicates.

**✨ Cleaning the rate Column:** This was a multi-step process:

Removed rows where rate was missing **(dropna)**.

Filtered out rows where rate was 'NEW' or '-'.

**Converted the string "4.1/5" into the number 4.1.**

# ✨ Cleaning Other Columns:

For cost2plates, we **filled missing values with the average cost**.

For rest_type and cuisines, we **filled missing values with the most common value (the mode)**.

**🗑️ Handling Duplicates (Round 2):** We run **df.drop_duplicates()** a final time to catch any new duplicates created during the cleaning process.

**🎯 Creating the Target Variable (Binning):** We define a function classify_rating that converts the numerical rate into three categories: 'Low', 'Average', or 'High'. This creates our final target column, rating_category.

# **4. 🤖 Final Data Prep for Machine Learning**
Before training, we perform **three final preparation steps:**

**Feature/Target Separation**: We split our DataFrame into X (the features) and y (the target).

**🔡 Encoding:** We loop through all text columns in X and use LabelEncoder to convert them into numbers.

**🔪 Train-Test Split:** We use train_test_split to divide our data. stratify=y is used to ensure both training and testing sets have a similar proportion of each rating category.

# **5. 🎨 Building the User Interface**
This section creates the visible parts of our app.

**Navigation:** We use **st.sidebar.radio()** to create a navigator that allows the user to switch between our two pages.

**if/elif Block:** The main logic that displays either Page 1 or Page 2 based on the user's choice.

# **6. 📈 Page 1: Individual Model Performance**
This page allows for a deep dive into a single model.

A **selectbox** in the sidebar lets the user choose one of the four models.

**The code then trains that model and displays two key metrics:**

**Classification Report:** A detailed table showing precision, recall, and f1-score.

**Confusion Matrix:** A heatmap that visually shows where the model made correct and incorrect predictions.

# **7. 🏆 Page 2: Model Comparison & Summary**
This page provides a high-level overview and a final recommendation.

It calls our **train_all_models()** function to get the accuracy scores for all four models.

It finds the model with the **highest score****.

It uses a helper function, **create_accuracy_circle**, to visualize the accuracy of each model in a cool donut-chart style.

Finally, it displays a summary using **st.success()** that declares the best model and explains why it was chosen.
