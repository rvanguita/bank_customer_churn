import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
            else:
                sns.boxplot(data=self.df, x=feature, color=self.color, orient='h', ax=ax)
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            self._customize_ax(ax)

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