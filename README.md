# Bank Customer Churn

## Introduction

The acquisition of new customers is always associated with a significant financial investment on the part of the company. Therefore, it is essential to avoid losing any customers and to identify the motivations behind their departure. In this project, an analysis of the data from a bank located in Europe was conducted to identify these motivations. Additionally, a machine learning classification model was used to predict which customers are most likely to leave the bank.

## Objective

The objective of this project is to analyze and develop the training of a machine learning model to identify which customers have a higher tendency to churn. A comparison was also made between different classification methods (CatBoostClassifier, LGBMClassifier, and XGBClassifier) using the following validation metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC, Matthews Correlation Coefficient, Cohen Kappa, and Log Loss.


### Repository Layout

The file main.ipynb presents all the code and other analyses performed on the data. In the assets/img folder, you can find all the images used in this document. In the data/ folder, you will find the zip file with the original data and the six CSV files representing the turbines.

## [Data Set](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

The dataset was obtained from Kaggle, where a variety of information about it is available, including descriptions of the meaning of each column and the types of data we might encounter, such as categorical, numerical, and so on. This information can be found in the main.ipynb file, right at the beginning of the document. I will not include this information here in the README.md to avoid cluttering the content.

## Methodology and Results


## Conclusion

## Criar e Ativar o Ambiente Virtual

1. Crie o ambiente virtual:
    ```bash
    python3 -m venv .venv
    ```

2. Ative o ambiente virtual:
    ```bash
    source .venv/bin/activate
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```