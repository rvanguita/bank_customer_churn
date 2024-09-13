import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import optuna


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    log_loss, matthews_corrcoef, cohen_kappa_score, roc_curve, auc
)

from sklearn.metrics import confusion_matrix



class ClassificationHyperTuner:
    def __init__(self, X_train, y_train, X_test, y_test, n_trials=30, model_name="catboost"):
        # Initialize data and configuration
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_trials = n_trials
        self.model_name = model_name

    def objective(self, trial):
        # Select model based on the given model name
        if self.model_name == "catboost":
            return self.boost_cat(trial)
        elif self.model_name == "lightgbm":
            return self.boost_lightgbm(trial)
        elif self.model_name == "xgboost":
            return self.boost_xg(trial)
        else:
            raise ValueError("Unsupported model. Choose between 'catboost', 'lightgbm', or 'xgboost'.")

    def run_optimization(self):
        # Study for hyperparameter optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best AUC: {study.best_value}")
        
        return study.best_params, study.best_value

    def boost_cat(self, trial):
        # Define hyperparameters for CatBoost
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        }

        model = cb.CatBoostClassifier(**params, silent=True)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, predictions)
        return auc

    def boost_lightgbm(self, trial):
        # Define hyperparameters for LightGBM
        params = {
            "objective": "binary",
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, predictions)
        return auc

    def boost_xg(self, trial):
        # Define hyperparameters for XGBoost
        params = {
            'objective': 'binary:logistic',
            'n_estimators': 5000,
            'seed': 42,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.05, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.05, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20)
        }

        model = xgb.XGBClassifier(**params)
        model.fit(self.X_train, self.y_train, verbose=0)
        predictions = model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, predictions)
        return auc


class ValidationClassification:
    def __init__(self, model, rouc_curve=False, oversampling=False, confusion_matrix=False):
        self.rouc_curve = rouc_curve
        self.oversampling = oversampling
        self.model = model
        self.confusion_matrix = confusion_matrix


    def plot_roc_curve(self, fpr, tpr, roc_auc, figsize=(4, 3), color='#c53b53'):
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color=color, lw=4, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Linha diagonal
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve', fontsize=16, weight='bold')
        plt.legend(loc="lower right")
        

        plt.gca().grid(False)
        plt.gca().yaxis.set_visible(True)
        plt.gca().xaxis.set_visible(True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        plt.show()


    # def plot_confusion_matrix(self, y, predictions):
    #     cm = confusion_matrix(y, predictions)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
    #     disp.plot(cmap='Blues', values_format='d')
    #     plt.show()


    def calculate_metrics(self, y, predictions, predictions_proba):
        metrics = {
            'Accuracy': accuracy_score(y, predictions),
            'Precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'Recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'F1 Score': f1_score(y, predictions, average='weighted', zero_division=0),
            'ROC AUC': roc_auc_score(y, predictions_proba[:, 1]),
            'Matthews Corrcoef': matthews_corrcoef(y, predictions),
            'Cohen Kappa': cohen_kappa_score(y, predictions),
            'Log Loss': log_loss(y, predictions_proba)
        }
        return {k: round(v * 100, 2) if k != 'Matthews Corrcoef' and k != 'Cohen Kappa' else round(v, 2) 
                for k, v in metrics.items()}


    def normal(self, X, y):
        self.model.fit(X, y)
        predictions_proba = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        scores = self.calculate_metrics(y, predictions, predictions_proba)
        scores_df = pd.DataFrame([scores])
        
        if self.confusion_matrix:
            print("Confusion Matrix:\n", confusion_matrix(y, predictions))
            # ValidationClassification.plot_confusion_matrix(y, predictions, model)
        
        if self.rouc_curve:

            fpr, tpr, _ = roc_curve(y, predictions_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            self.plot_roc_curve(fpr, tpr, roc_auc)

        return scores_df


    def cross(self, X, y, n_splits=5):
        cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        metrics_cross = {key: [] for key in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 
                                             'Matthews Corrcoef', 'Cohen Kappa', 'Log Loss']}
        confusion_matrices = []
        roc_auc_scores = []
        fpr_list = []
        tpr_list = []
        
        all_predictions = []
        all_true_labels = []

        for index_train, index_validation in cv.split(X, y):
            X_train, X_validation = X[index_train], X[index_validation]
            y_train, y_validation = y[index_train], y[index_validation]

            if self.oversampling:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_validation)
            predict_proba = self.model.predict_proba(X_validation)
            
            metrics = self.calculate_metrics(y_validation, predictions, predict_proba)
            for key in metrics_cross.keys():
                metrics_cross[key].append(metrics[key])

            if self.confusion_matrix:
                # cm = confusion_matrix(y_validation, predictions)
                # confusion_matrices.append(cm)
                
                all_predictions.extend(predictions)
                all_true_labels.extend(y_validation)
                
                
            if self.rouc_curve:    
                fpr, tpr, _ = roc_curve(y_validation, predict_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                roc_auc_scores.append(roc_auc)
                fpr_list.append(fpr)
                tpr_list.append(tpr)

        if self.confusion_matrix:
            final_confusion_matrix = confusion_matrix(all_true_labels, all_predictions)
            print("Final Confusion Matrix (All Folds):\n", final_confusion_matrix)
            # self.plot_confusion_matrix(np.array(all_true_labels), np.array(all_predictions))


        if self.rouc_curve:
            self.plot_roc_curve(fpr_list[-1], tpr_list[-1], roc_auc_scores[-1])




        scores = {key: round(np.mean(val), 2) if key != 'Matthews Corrcoef' and key != 'Cohen Kappa' 
                  else round(np.mean(val), 2) for key, val in metrics_cross.items()}
        scores_df = pd.DataFrame([scores])
        
        return scores_df


class DataVisualizer:
    def __init__(self, df, color='#c3e88d', figsize=(24, 12)):
        """
        Initializes the DataVisualizer with a DataFrame and default plot settings.

        Args:
        - df (pd.DataFrame): DataFrame containing the data.
        - color (str, optional): Default color for the plots.
        - figsize (tuple, optional): Default figure size.
        """
        self.df = df
        self.color = color
        self.figsize = figsize

    def plot_barplot(self, features, hue=None, custom_palette=None):
        rows, cols = self._calculate_grid(len(features))
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            grouped = self.df.groupby([feature, hue]).size().reset_index(name='count')
            total_count = grouped['count'].sum()
            grouped['percentage'] = (grouped['count'] / total_count) * 100

            num_categories = self.df[feature].nunique()
            width = 0.8 if num_categories <= 5 else 0.6

            sns.barplot(data=grouped, x='count', y=feature, palette=custom_palette, hue=hue, ax=ax, width=width, orient='h')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            self._customize_ax(ax)
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            ax.yaxis.set_visible(True)
            ax.xaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)
            
            for p in ax.patches:
                width = p.get_width()
                percentage = (width / total_count) * 100
                if percentage != 0:
                    ax.annotate(f'{percentage:.1f}%', 
                                (width, p.get_y() + p.get_height() / 2),
                                xytext=(5, 0), 
                                textcoords="offset points",
                                ha='left', va='center',
                                fontsize=11, color='black', fontweight='bold')


        self._remove_extra_axes(axes, len(features))
        plt.tight_layout()

    def plot_boxplot(self, features, hue=None, custom_palette=None):
        rows, cols = self._calculate_grid(len(features))
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            if hue:
                sns.boxplot(data=self.df, x=feature, y=hue, hue=hue, orient='h', palette=custom_palette, ax=ax)
                ax.set_ylabel('')
            else:
                sns.boxplot(data=self.df, x=feature, color=self.color, orient='h', ax=ax)
                ax.yaxis.set_visible(False)
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            self._customize_ax(ax)

            ax.set_xlabel('')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)

        self._remove_extra_axes(axes, len(features))
        plt.tight_layout()

    def plot_histplot(self, features, hue=None, custom_palette=None, kde=False):
        rows, cols = self._calculate_grid(len(features))
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            sns.histplot(data=self.df, x=feature, hue=hue, palette=custom_palette, kde=kde, ax=ax, stat='proportion')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            self._customize_ax(ax)
            
            ax.set_xlabel('')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            ax.yaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)
            
        self._remove_extra_axes(axes, len(features))
        plt.tight_layout()

    def plot_seaborn_bar(self, hue, custom_palette):
        """
        Plots a bar chart displaying the percentage distribution of a categorical variable.
        """
        percentage_exited = self.df[hue].value_counts(normalize=True) * 100
        percentage_df = percentage_exited.reset_index()
        percentage_df.columns = [hue, 'percentage']

        plt.figure(figsize=(4, 3))
        ax = sns.barplot(data=percentage_df, x=hue, y='percentage', hue=hue, palette=custom_palette)
        
        ax.set_title(f'{percentage_exited.idxmin()} rate is about {percentage_exited.min():.2f}%', 
                     fontweight='bold', fontsize=13, pad=15, loc='center')
        ax.set_xlabel('')
        ax.invert_xaxis()
        ax.tick_params(axis='both', which='both', length=0)
        ax.yaxis.set_visible(False)
        self._customize_ax(ax, hide_spines=True)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}%', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, -10), 
                        textcoords='offset points',
                        fontsize=11, color='white', fontweight='bold')

        plt.tight_layout()

    def plot_custom_scatterplot(self, x, y, hue, custom_palette, title_fontsize=16, label_fontsize=14):
        """
        Plots a customized scatter plot with specified x and y axes, hue for color coding, and custom palette.
        """
        plt.figure(figsize=(16, 6))
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue, palette=custom_palette, alpha=0.6)

        plt.title(f'{x} vs {y}', fontsize=title_fontsize, weight='bold')
        plt.xlabel(x, fontsize=label_fontsize)
        plt.ylabel(y, fontsize=label_fontsize)

        ax = plt.gca()
        self._customize_ax(ax, hide_spines=True)

        legend = ax.get_legend()
        if legend:
            legend.set_title(hue)
            legend.set_bbox_to_anchor((1.15, 0.8))

    def _calculate_grid(self, num_features):
        rows = (num_features + 2) // 3
        cols = min(3, num_features)
        return rows, cols

    def _customize_ax(self, ax, hide_spines=False):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if hide_spines:
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        ax.grid(False)

    def _remove_extra_axes(self, axes, num_features):
        for j in range(num_features, len(axes)):
            plt.delaxes(axes[j])