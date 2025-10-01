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



#* ----------------------------------------------------------------------------
#*                            PAGE CONFIGURATION
#* ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Zomato Rating Predictor",
   page_icon="🍔",
   layout="wide",
   initial_sidebar_state="expanded"
)


#* ----------------------------------------------------------------------------
#*              DATA PREPARATION -->    DATA LOADING AND PREPROCESSING
#* ----------------------------------------------------------------------------

@st.cache_data  #this tell the function to run only once, "Magic command"
#once the function runs, it stores the result in the memory and reuses it, makingn the app incredibly fast
def load_and_preprocess_data():
    #Load the full dataset from the CSV file
    df=pd.read_csv('zomato.csv')

    #Take a random sample of 10,000 rows to keep the app fast.
    df_sample=df.sample(n=10000, random_state=42)

    #Rename a tricky column for easier use.
    df_sample.rename(columns={'approx_cost(for two people)':'cost2plates'}, inplace=True)


    #select onlly the columns we need for our analysis
    columns_to_keep=[
        'online_order',
        'rate',
        'votes',
        'location',
        'rest_type',
        'cuisines',
        'cost2plates',
        'listed_in(type)'
    ]

    df_clean=df_sample[columns_to_keep].copy()


    #* ----------------------------------------------------
    #*                  DATA CLEANING
    #* ----------------------------------------------------
    #1.Handle duplicate records
    df_clean.drop_duplicates(inplace=True)

    #2.Handle the 'rate' column: missing values and data type conversion
    df_clean.dropna(subset=['rate'], inplace=True)
    df_clean=df_clean[df_clean['rate']!='NEW']

    df_clean = df_clean[df_clean['rate'] != '-']


    df_clean['rate']=df_clean['rate'].apply(lambda x: float (x.split('/')[0]))
    
    #3.Handle the 'cost2plates' column: data type and missing values with mean (as more no of values are missing)
    df_clean['cost2plates']=df_clean['cost2plates'].astype(str).str.replace(',','').astype(float)
    mean_cost=df_clean['cost2plates'].mean()
    df_clean['cost2plates'].fillna(mean_cost, inplace=True)

    #4.Handle other columns with a few missing values with their mode (as less values are missing)
    df_clean['rest_type'].fillna(df_clean['rest_type'].mode()[0], inplace=True)
    
    #************ here i have encountered after cleaning the data that we still had duplicate records it's because
    #*** when we replace the missing values like nan or null values with mean or mode then 
    #** we r creating duplicates by filling those so we need to drop duplicates again
    df_clean['cuisines'].fillna(df_clean['cuisines'].mode()[0], inplace=True)

    #** here we are droping the duplicate values
    df_clean.drop_duplicates(inplace=True)

    #* ----------------------------------------------------
    # *             CREATE TARGET VARIABLE (DATA BINNING)
    #* ----------------------------------------------------
    #define a function to classify rating into categories.
    def classify_rating(rate):
        if rate>=4.0:
            return 'High'
        elif rate>=3.0:
            return 'Average'
        else:
            return 'Low'
        
    #Apply this function to the 'rate' column to create our new target column.
    df_clean['rating_category']=df_clean['rate'].apply(classify_rating)

    return df_clean

# ----------------------------------------------------------------------------
#                            EXPLORE THE DATA
# ----------------------------------------------------------------------------
# This is a temporary section to diagnose our data.
# We will remove or comment this out later

#--------------------------------------------------
        #this was done during the verification step for duplicate records
#-------------------------------------------------
#Load the data by calling our function
#^
#|
#|
# *df_clean=load_and_preprocess_data()

# *#Display a subheader in the app.
# st.subheader("1. Initial Data Exploration")
# st.write("Let's check for missing values and duplicates in our sampled and cleaned dataset.")

#* #Check for missing values.
#* st.write("Missing Values Count:")
#* missing_values=df_clean.isnull().sum()
#* st.write(missing_values)

#* #Check for duplicate records.
#* st.write("Duplicate Records Count: ")
#* duplicate_count=df_clean.duplicated().sum()
# *st.write(f"We found {duplicate_count} duplicate rows.")

# ----------------------------------------------------------------------------
#                            VERIFY THE CLEANING 
# ----------------------------------------------------------------------------
# This is a temporary section to verify our data cleaning.
# We will remove this later when we build the final UI.

# #*Load the data by calling our function.
#*df_processed = load_and_preprocess_data()

# Display a subheader in the app.
#*st.subheader("2. Data Cleaning Verification")
#*st.write("Let's check the state of our data after cleaning.")

# Check for missing values again.
#*st.write("Missing Values After Cleaning:")
#*st.write(df_processed.isnull().sum())

# Check for duplicate records again.
#*st.write("Duplicate Records After Cleaning:")
#*duplicate_count_after = df_processed.duplicated().sum()
#*st.write(f"We found {duplicate_count_after} duplicate rows.")

# Check the final size of our dataset.
#*st.write("Final Shape of the Dataset:")
#*st.write(df_processed.shape)



#* ----------------------------------------------------------------------------
#*                   FINAL DATA PREP FOR MACHINE LEARNING
#* ----------------------------------------------------------------------------

#Load the final, clean data
df=load_and_preprocess_data()

# Separate featurees (X) and the target variable (y).
X=df.drop('rating_category', axis=1)
y=df['rating_category']

#Encode all categorical features (text columns) into numbers
# We create a dictionary to store the encoders for potential future use.
encoders={}
for col in X.select_dtypes(include='object').columns:
    le=LabelEncoder()
    X[col]=le.fit_transform(X[col])
    encoders[col]=le

#Split the data into training and testing sets.
X_train,X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)




#* ----------------------------------------------------------------------------
#*                            BUILD THE USER INTERFACE
#* ----------------------------------------------------------------------------
#Main title and description
st.title("🍔 Zomato Restaurant Rating Predictor")
