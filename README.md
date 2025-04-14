# Bank Customer Churn

![](assets/img/cape.webp)

## Introduction

The acquisition of new customers is always associated with a significant financial investment on the part of the company. Therefore, it is essential to avoid losing any customers and to identify the motivations behind their departure. In this project, an analysis of the data from a bank located in Europe was conducted to identify these motivations. Additionally, a machine learning classification model was used to predict which customers are most likely to leave the bank.

## Objective

The objective of this project is to analyze and develop the training of a machine learning model to identify which customers have a higher tendency to churn. A comparison was also made between different classification methods (catboost, lightgbm, and xgboost) using the following validation metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC, Matthews Correlation Coefficient, Cohen Kappa, and Log Loss.


### Repository Layout

The file **__main.ipynb__** presents all the code and other analyses performed on the data. In the **__assets/img__** folder, you can find all the images used in this document. In the **__data/__** folder, you will find the zip file with the original data and the six CSV files representing the turbines. The **__requirements.txt__** file is where all the libraries used in this project are listed

## [Data Set](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

The dataset was obtained from Kaggle, where a variety of information about it is available, including descriptions of the meaning of each column and the types of data we might encounter, such as categorical, numerical, and so on. This information can be found in the **main.ipynb** file, right at the beginning of the document. I will not include this information here in the README.md to avoid cluttering the content.

## Methodology and Results

After conducting an initial analysis to identify possible duplicate rows, incorrect variable types in the DataFrame, and missing rows, it was possible to determine the proportion of customers who chose to leave. The figure below illustrates this percentage, revealing that **20.38%** of customers opted to leave the bank, pertaining to the period during which the database was developed.

![](assets/img/1.png)

Below is another bar chart analyzing each item in relation to customers who left the bank (red) or stayed (blue). The values analyzed include: Gender, Geography, Card Type, NumOfProducts, HasCrCard, IsActiveMember, Complain, and Satisfaction Score. It is possible to observe that most of this data does not have a significant impact on the number of customers who left the service. In the case of Complaints, we have a different analysis: all individuals who made a complaint left the bank, and only 0.**1%** of the customers who complained remained with the institution. For the purposes of model training, this item was not used, as the goal of this work is to develop training with various classification models. On the other hand, from a business model perspective, this is quite concerning. It is highly likely that this company has a customer support team facing serious issues, and measures should be taken to avoid such situations.

![](assets/img/2.png)

This entire work uses the same color code described in the paragraph above. In the chart below, it is possible to observe six histograms of the following topics: CreditScore, Age, Balance, Estimated Salary, Points Earned, and Tenure. The only one that shows a different average value among customers who left the company is Age.

![](assets/img/3.png)

Below is a scatter plot showing the relationship between balance and age. There is a predominance of customers who left the company in the age range of 50 to 65 years, regardless of their account balance.

![](assets/img/4.png)

A boxplot was also created using the following data: CreditScore, Age, Balance, NumOfProducts, EstimatedSalary, Satisfaction Score, and Points Earned. This plot was used to identify potential outliers and assess if any of these values were more prevalent among customers who left or remained with the bank. For this project, this was an initial exploratory analysis, with no actions taken regarding the outliers found.

![](assets/img/5.png)

![](assets/img/6.png)

Before starting the tests with the three classification models, several modifications and additions were made to the data. As part of feature engineering, a new column was created with grouped age ranges as follows: '0-24' < '25-34' < '35-44' < '45-54' < '55-64' < '65+'. The reason for this was to help the models more easily identify which age groups have a higher likelihood of leaving the bank. Another column was also created to categorize each customer's balance as 'Negative' < 'Low' < 'Medium' < 'High', once again making it easier for the model to identify customers with higher or lower balance values. Both of these variables were encoded using the OrdinalEncoder, along with the card type.

By analyzing the data, a possible trend was identified between age, balance, and the likelihood of customers leaving the bank. For this reason, these two variables were passed through PolynomialFeatures, generating a second-degree polynomial and creating three new columns: Age^2, Age*Balance, and Balance^2.

Additionally, the StandardScaler was applied to the following columns: 'CreditScore', 'Balance', 'EstimatedSalary', and 'Point Earned'. Finally, the OneHotEncoder was used for the columns: 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Satisfaction Score', and 'Geography'.

At the end of this process, I obtained a DataFrame with the following columns: 'AgeGroup', 'BalanceCategory', 'Age', 'Balance', 'Age^2', 'Age Balance', 'Balance^2', 'CreditScore','EstimatedSalary', 'Point Earned', 'Card_Type_OrdinalEncoder', 'BalanceCategory_Encoded', 'AgeGroup_Encoded', 'Gender_Male', 'NumOfProducts_2', 'NumOfProducts_3', 'NumOfProducts_4', 'HasCrCard_1', 'IsActiveMember_1', 'Satisfaction Score_2', 'Satisfaction Score_3', 'Satisfaction Score_4', 'Satisfaction Score_5', 'Geography_Germany', 'Geography_Spain'.

With the DataFrame data prepared, the classification models CatBoost, LightGBM, and XGBoost were applied. For each model, a simplified optimization of their hyperparameters was performed, as more detailed adjustments or deeper searches resulted in overfitting and worsened the validation parameters. The table below presents the results of the analysis, where each model was evaluated using data splitting into training and test sets with the train_test_split function, as well as cross-validation with the KFold function. The table is sorted in descending order based on the Log Loss metric.


| Model        | Accuracy | Precision | Recall | F1 Score | ROC AUC | Matthews Corrcoef  | Cohen Kappa | Log Loss |
|:-------------|:--------:|:---------:|:------:|:--------:|:-------:|:------------------:|:-----------:|:--------:|
| normal_lgb   |  94.60   |   94.62   |  94.60 |  94.39   |  99.02  |        0.83        |     0.82    |  16.96   |
| normal_cb    |  89.35   |   89.12   |  89.35 |  88.47   |  93.42  |        0.65        |     0.63    |  26.05   |
| normal_xgb   |  89.45   |   89.15   |  89.45 |  88.68   |  92.73  |        0.65        |     0.63    |  26.70   |
| cross_xgb    |  86.57   |   85.75   |  86.57 |  85.43   |  86.81  |        0.54        |     0.53    |  33.21   |
| cross_cb     |  86.59   |   85.78   |  86.59 |  85.44   |  86.95  |        0.54        |     0.53    |  32.95   |
| cross_lgb    |  86.12   |   85.17   |  86.12 |  84.98   |  85.91  |        0.53        |     0.51    |  34.31   |



The best-performing method was normal_lgb, but due to its high accuracy, there are indications that it might be overfitting. Therefore, the second-best method was chosen instead.

When analyzing the normal_cb and normal_xgb models, we observed that both produced very similar results, with normal_cb achieving a slightly lower Log Loss compared to normal_xgb. However, when comparing the other metrics — Accuracy, Precision, Recall, and F1 Score — the normal_xgb method outperformed its competitor.

Thus, the selected method was normal_xgb, considering that the difference in Log Loss between it and normal_cb was minimal, and that out of the eight evaluated metrics, it outperformed in four, while the remaining four showed very similar or identical values.

With the selected model, the simulation was performed using the test data. Below are the confusion matrix and the ROC curve. A table with the system's validation results is also presented. The data shows that the application of this method to the problem at hand yields good metrics, particularly: Accuracy, Precision, Recall, F1 Score, ROC AUC, and the ROC curve, all with values close to 90. The Matthews Corrcoef and Cohen Kappa metrics also showed significant results, with values above 0.6, while Log Loss was very close to zero.


![](assets/img/7.png)


![](assets/img/8.png)


| Accuracy | Precision | Recall | F1 Score | ROC AUC | Matthews Corrcoef | Cohen Kappa | Log Loss |
|:--------:|:---------:|:------:|:--------:|:-------:|:-----------------:|:-----------:|:--------:|
|  89.84   |   89.51   |  89.84 |   89.11  |  93.05  |       0.66        |     0.64    |  26.08   |


To extract more information from this analysis, the SHAP library was used to identify which features had the greatest impact on the model's predictions. As noted earlier, age had a significant impact on customer classification, with older individuals showing greater differentiation. Another factor that had not been highlighted was the presence of customers who have product number 2. Since this table is binary and there is a large predominance of people who have this product and have not churned, this information becomes relevant.

![](assets/img/9.png)


![](assets/img/10.png)

## Conclusion

This work aimed to implement an analysis on the database of a European bank, with the purpose of training a machine learning model to predict which customers will leave the institution and to identify the possible factors contributing to this decision. Early in the project, a glaring issue was identified: all customers who made complaints ended up leaving the company. From the perspective of this study, this column presented a significant bias, as it made it easy to identify customers who were likely to churn. Therefore, this data was excluded from the analysis. Additionally, it is recommended that the company invest in a more in-depth analysis across all departments related to problem resolution to investigate what those problems might be.

After processing the data, a DataFrame was created with the information described in the methodology, and six tests were conducted. The results obtained are nearly perfect, in contrast to the issues often encountered in the real world. It is possible that the creator of this dataset introduced some bias, resulting in models with performance so close to perfection. With these caveats in mind, XGBoost showed the best results across all evaluated parameters. At the end of this project, it was also possible to identify which features were most relevant for classification.