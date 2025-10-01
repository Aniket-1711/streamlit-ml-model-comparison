#core libraries for App and Data Handling
import streamlit as st
import pandas as pd 
import numpy as np

#Libraries for data Visualization
import seaborn as sns
import matplotlib.pyplot as plt


#Libraries from Scikit - Learn for machine learning --> use pip install scikit-learn while installing in terminal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


#importing model 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



# ----------------------------------------------------------------------------
#                            PAGE CONFIGURATION
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Zomato Rating Predictor",
   page_icon="🍔",
   layout="wide",
   initial_sidebar_state="expanded"
)


# ----------------------------------------------------------------------------
#                      DATA LOADING AND PREPROCESSING
# ----------------------------------------------------------------------------

@st.cache_data  #this tell the function to run only once, "Magic command"
#once the function runs, it stores the result in the memory and reuses it, makingn the app incredibly fast
def load_and_preprocess_data():
    #Load the full dataset from the CSV file
    df=pd.read_csv('zomato.csv')

    #Take a random sample of 10,000 rows to keep the app fast.
    df_sample=df.sample(n=10000, random_state=42)

    #select onlly the columns we need for our analysis
    columns_to_keep=[
        'online_order',
        'rate',
        'votes',
        'location',
        'rest_type',
        'cuisines',
        'approx_cost(for two people)',
        'listed_in(type)'
    ]

    df_clean=df_sample[columns_to_keep].copy()

    #Rename a tricky column for easier use.
    df_clean.rename(columns={'approx_cost(for two poeple)':'cost2plates'}, inplace=True)

     # ----------------------------------------------------
    #                  DATA CLEANING
    # ----------------------------------------------------

    
    return df_clean

# ----------------------------------------------------------------------------
#                            EXPLORE THE DATA
# ----------------------------------------------------------------------------
# This is a temporary section to diagnose our data.
# We will remove or comment this out later

#Load the data by calling our function
df_clean=load_and_preprocess_data()

#Display a subheader in the app.
st.subheader("1. Initial Data Exploration")
st.write("Let's check for missing values and duplicates in our sampled and cleaned dataset.")

#Check for missing values.
st.write("Missing Values Count:")
missing_values=df_clean.isnull().sum()
st.write(missing_values)

#Check for duplicate records.
st.write("Duplicate Records Count: ")
duplicate_count=df_clean.duplicated().sum()
st.write(f"We found {duplicate_count} duplicate rows.")

