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
* Classes [Customer will EXIT(1) or NOT(0)] are properly balanced


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




1] Did some EDA for finding relationship between columns

2] Processed the data

3] trained the data on machine learning model

4] evaluated to check model's performance on unseen data

Mlops practise
---

1] Created a DVC pipeline
   - pipeline to execute processing, training and evaluating
   - using DVC for tracking the version of data
     
2] Mlflow
   - For experiment tracking
