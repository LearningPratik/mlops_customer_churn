# Customer Churn project 
---

Churn prediction, or the task of identifying customers who are likely to discontinue use of a service, is an important and lucrative concern of any industry.

Description

This project is tasked to predict the churn score using features such as:
* Historical purchase data
* Tenure, charges and other information
    
This project aims to identify customers who are likely to leave so that we can retain them with certain incentives.

# Dataset:

* Dataset has been taken from Kaggle, and raw dataset can be downloaded from here. https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* Classes [Customer will churn or not]


# Notebook contains the EDA, data processing
https://github.com/LearningPratik/mlops_customer_churn/tree/main/notebooks

# DVC pipeline : 
https://github.com/LearningPratik/mlops_customer_churn/tree/main/src 
* This link contains .py file for --> data collection, data processing, model training model evaluation

# Models

 Why this model ? : to get started and Random Forest is robust, in further experiments plan is to try out some other models
 * Random Forest

# Project Pipeline :

![image](https://github.com/user-attachments/assets/60748462-e884-44e3-a621-c4032ea21b30)

![image](https://github.com/user-attachments/assets/b9f1fa5c-7aec-4ac9-a95f-3b12dd6e9b95)


# Techstack
Python version : 3.10 <br>
Packages: pandas, numpy, sklearn, fastapi, seaborn <br>
MLops tools: DVC (pipeline), MLFlow (experiment tracking)


# Results and observation
* Churn column, which is our very unbalanced (planning to use resampling techniques)
* some features are with 3 categories and pattern for this categories is (2 are balanced and 1 is either higher or lower in number)
* ex for above are --> contract, streamingmovies, payment methods
* there are only 2 numerical columns and other are categorical columns
