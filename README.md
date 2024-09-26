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

Balance, Age


![](assets/img/4.png)

CreditScore, Age, Balance, NumOfProducts, EstimatedSalary, Satisfaction Score, Point Earned

![](assets/img/5.png)

![](assets/img/6.png)



| Model        | Accuracy | Precision | Recall | F1 Score | ROC AUC | Matthews Corrcoef | Cohen Kappa | Log Loss |
|:-------------|:--------:|:---------:|:------:|:--------:|:-------:|:-----------------:|:-----------:|:--------:|
| normal_xgb   |  99.99   |   99.99   |  99.99 |  99.99   |  100.00 |        1.00       |     1.00    |  3.89    |
| normal_lgb   |  99.33   |   99.34   |  99.33 |  99.33   |  99.98  |        0.98       |     0.98    |  9.60    |
| normal_cb    |  98.01   |   98.05   |  98.01 |  97.98   |  99.77  |        0.94       |     0.94    |  10.20   |
| cross_cb     |  85.98   |   85.07   |  85.98 |  84.98   |  85.43  |        0.53       |     0.51    |  35.72   |
| cross_lgb    |  85.72   |   84.69   |  85.72 |  84.64   |  85.42  |        0.51       |     0.50    |  36.07   |
| cross_xgb    |  85.45   |   84.51   |  85.45 |  84.53   |  83.76  |        0.51       |     0.50    |  42.26   |


![](assets/img/7.png)


![](assets/img/8.png)









| Accuracy | Precision | Recall | F1 Score | ROC AUC | Matthews Corrcoef | Cohen Kappa | Log Loss |
|:--------:|:---------:|:------:|:--------:|:-------:|:-----------------:|:-----------:|:--------:|
|  100.0   |   100.0   |  100.0 |   100.0  |  100.0  |        1.0        |     1.0     |  1.08    |

![](assets/img/9.png)


![](assets/img/10.png)

## Conclusion