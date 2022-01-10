# Statistical_Comparison_KAGGLE_Telco_Churn
A toy project where I try a statistical comparison technique (5x2 cross_validation) to compare two models trained on a very basic telco churn task. 

# Description 

In this toy project, I use the dataset [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) from Kaggle to try a statistical comparison technique. 
Basically, I train one SVM and one Random Forest, I quickly hypertune them using GridSearch (and saving result in an mlflow run), and I then compare them using 5x2 cross-validation.

Simply install requirements using pip and run ``` python main.py ```  
