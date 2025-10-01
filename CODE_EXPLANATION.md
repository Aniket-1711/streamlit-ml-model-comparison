# 🍽️ Zomato Rating Predictor  

An interactive **Streamlit web application** that predicts restaurant ratings using multiple machine learning models.  

---

## 📚 1. Importing Libraries  

We start by importing all the necessary libraries:  

- **streamlit 🌐** → Build interactive web apps.  
- **pandas 🐼** → Data handling and cleaning.  
- **numpy 🔢** → Numerical computations.  
- **seaborn & matplotlib 📊** → Data visualization (Confusion Matrix, plots).  
- **scikit-learn 🧠** → Machine learning toolbox.  
  - `train_test_split` 🔪 → Splits data into training & testing sets.  
  - `LabelEncoder` 🏷️ → Converts text → numbers.  
  - `classification_report`, `confusion_matrix`, `accuracy_score` 📜 → Model evaluation metrics.  
  - Models used: `LogisticRegression`, `KNeighborsClassifier`, `SVC`, `GaussianNB`.  

---

## 📄 2. Page Configuration  

We configure the Streamlit app using `st.set_page_config()`:  

- **page_title** → Title shown on browser tab.  
- **page_icon 🍔** → Small favicon.  
- **layout = "wide"** → Full browser width.  
- **initial_sidebar_state = "expanded"** → Sidebar open by default.  

---

## 🧹 3. Data Loading & Preprocessing  

Defined in `load_and_preprocess_data()` function with `@st.cache_data` to improve performance.  

### Cleaning Steps:  

1. **📥 Load Data** → Read `zomato.csv`.  
2. **🎲 Sampling** → Take 10,000 random rows (`random_state=42`).  
3. **✏️ Rename Columns** → e.g., `approx_cost(for two people)` → `cost2plates`.  
4. **🗑️ Duplicates (Round 1)** → Remove duplicates.  
5. **✨ Clean `rate` column** →  
   - Drop missing, "NEW", and "-" values.  
   - Convert `"4.1/5"` → `4.1`.  
6. **✨ Clean Other Columns** →  
   - Fill missing `cost2plates` with average.  
   - Fill missing `rest_type` and `cuisines` with mode.  
7. **🗑️ Duplicates (Round 2)** → Final cleanup.  
8. **🎯 Target Variable** → Create `rating_category` (`Low`, `Average`, `High`).  

---

## 🤖 4. Final Prep for Machine Learning  

- **Feature/Target Separation** → Split `X` (features) & `y` (target).  
- **🔡 Encoding** → Convert categorical columns into numbers with `LabelEncoder`.  
- **🔪 Train-Test Split** → Use `train_test_split` with `stratify=y`.  

---

## 🎨 5. Building the User Interface  

- **Navigation** → Sidebar with `st.sidebar.radio()`.  
- **if/elif logic** → Switch between two pages.  

---

## 📈 6. Page 1: Individual Model Performance  

- Sidebar **selectbox** → Choose one of the four models.  
- Train & display:  
  - **Classification Report** → Precision, Recall, F1-score.  
  - **Confusion Matrix** → Visual performance analysis.  

---

## 🏆 7. Page 2: Model Comparison & Summary  

- Train all four models → Compare accuracies.  
- Find **best model** automatically.  
- Visualize accuracy using **donut charts**.  
- Display final recommendation with `st.success()`.  

---
