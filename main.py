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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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


#**Function to train all the models
@st.cache_data
def train_all_models(X_train, y_train, X_test, y_test):
    # A dictionary to hold the accuracy scores of each model.
    accuracies={}

    # Initialize all the models
    models={
        "Logistic Regression":LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors":KNeighborsClassifier(),
        "Support Vector Machine":SVC(),
        "Naive Bayes": GaussianNB()
    }


    #Loop through each model, train it, and store it's accuracy.
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        accuracies[name]=accuracy_score(y_test, y_pred)

    return accuracies
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

#Sidebar for navigation
st.sidebar.title("Navigation")
nav_choice=st.sidebar.radio("Go to:",
                            [
                                "Individual Model Performance",
                                "Model Comparison & Summary" 
])

#*-----------------------------------------------------------------------
#*------- Page 1: Individul Model performance ---------------------------
#*-----------------------------------------------------------------------
if nav_choice=="Individual Model Performance":
    st.title("🍔 Zomato Restaurant Rating Predictor")
    st.markdown("""
    This dashboard predicts the rating category of a restaurant in Bangalore
                based on its features. Choose a model from the sidebar to see its performance.
                """)
    
    #Sidebar for model selection
    st.sidebar.title("Model Selection")
    models={
        "Logistic Regression":LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors":KNeighborsClassifier(),
        "Support Vector Machine":SVC(),
        "Naive Bayes": GaussianNB()
    }

    selected_model=st.sidebar.selectbox("Select a Model", list(models.keys()))


    #* ----------------------------------------------------------------------------
    #*                      MODEL TRAINING AND PERFORMANCE
    #* ----------------------------------------------------------------------------
    #Display a header for this section, which changes based on the model selected.
    st.header(f"Performance of: {selected_model}")

    #Get the model object from our dicitionary
    model=models[selected_model]

    #Train the model on the training data.
    model.fit(X_train, y_train)

    #Make predictions on the unseen test data.
    y_pred=model.predict(X_test)


    #Get then names of our target categories (e.g 'Low', 'Average', 'High')
    target_names=sorted(df['rating_category'].unique())

    #*---Display Metrics-----

    # 1. Classification Report
    st.subheader("Classification Report")
    report=classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df=pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(2))


    #2. Confusion matrix
    st.subheader("Confusion Matrix")
    #Use columns to display the plot and explantion side-by-side
    col1, col2=st.columns([2,1])


    with col1:
        cm=confusion_matrix(y_test, y_pred)
        fig, ax=plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names,ax=ax)
        ax.set_xlabel('Predicted Lable')
        ax.set_ylabel('True Label')
        st.pyplot(fig)

    with col2:
        st.markdown("""
        **How to read the matrix:**
        - The **rows** represent the actual rating category.
        - The **columns** represent the category predicted the model.
        - The **diagonal numbers** show the number of correct predictions.
        - **Off-Diagonal numbers** show where the model made mistakes.
    """)
        
    st.info(f"The {selected_model} model was trained and evaluated. Check the report and matrix above to see its performance.")



#* ----------------------------------------------------------------------------
# *                      PAGE 2: MODEL COMPARISON & SUMMARY
#* ----------------------------------------------------------------------------
elif nav_choice=="Model Comparison & Summary":
    st.title("🏆 Model Comparison & Summary")
    st.markdown("Here we compare the accuracy of all models and recommend the best one.")


    #Get the accuracy scores by calling our function
    accuracies=train_all_models(X_train, y_train, X_test, y_test)

    #Find the best model
    best_model_name=max(accuracies, key=accuracies.get)
    best_accuracy=accuracies[best_model_name]

    #----- A helper function to draw the accuracy circle----
    def create_accuracy_circle(score, title):
        fig, ax=plt.subplots(figsize=(3,3))
        #use a pie chart to create the donut effect
        ax.pie([score, 1-score], startangle=90, colors=['#4CAF50', '#E0E0E0'],
               wedgeprops=dict(width=0.3, edgecolor='w'))
        
        # Add the percentage text in the center
        ax.text(0, 0, f"{score: .1%}", ha='center', va='center', fontsize=20, weight='bold')
        ax.set_title(title, fontsize=12)
        return fig

    #!Display the accuracy circles in a 2x2 grid

    st.subheader("Model Accuracy Comparison")
    col1, col2=st.columns(2)
    with col1:
        fig=create_accuracy_circle(accuracies["Logistic Regression"], "Logistic Regression")
        st.pyplot(fig)
        fig=create_accuracy_circle(accuracies["Support Vector Machine"], "Support Vector Machine")
        st.pyplot(fig)
    with col2:
        fig=create_accuracy_circle(accuracies["K-Nearest Neighbors"], "K-Nearest Neighbors")
        st.pyplot(fig)
        fig=create_accuracy_circle(accuracies["Naive Bayes"], "Naive Bayes")
        st.pyplot(fig)

    
    #Display the final recommendation
    st.subheader("Final Recommendation")
    st.success(f"""
         After comparing all four models, the **{best_model_name}** emerges as the top performer
    for this dataset with an accuracy of **{best_accuracy:.1%}**.

    This suggests that its approach to finding patterns in the data is the most effective
    for distinguishing between 'Low', 'Average', and 'High' rated restaurants based on the
    features we've provided.
    """)