# Customer Experience Dashboard

Name: Naman Srivastava  
Internship: AI Case Study  

---

## Project Overview
This project is about creating a Customer Experience Dashboard to study customer orders, deliveries, and feedback.  
It uses data analysis, sentiment analysis, and a machine learning model to understand customer satisfaction and find ways to improve it.

---

## Tools and Technologies
- Python 3.13  
- Pandas, NumPy, NLTK, Scikit-Learn, Matplotlib, Streamlit  
- Visual Studio Code

---

## Methodology
1. Combined different datasets (orders, feedback, delivery, etc.)  
2. Cleaned and processed the data using `data_processing.py`  
3. Used NLTK to analyze the sentiment of customer feedback  
4. Trained a Random Forest model in `train_model.py` to predict satisfaction  
5. Built a Streamlit app (`streamlit_app.py`) to show the results and insights  

---

## Results
- Model Accuracy: 98%  
- ROC-AUC Score: 1.0  
- The dashboard shows clear insights about delivery performance, order trends, and customer satisfaction.  

---

## How to Run
1. Create a virtual environment:
   python -m venv venv
   .\venv\Scripts\activate

2. pip install -r requirements.txt

3. streamlit run app/streamlit_app.py