# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_seaborn_bar(df, custom_palette, hue):
    """
    Plots a bar chart displaying the percentage distribution of a categorical variable.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
        This is the source dataset from which the percentages of the `hue` variable will be calculated.

    - custom_palette (dict): A custom color palette for the bars.
        This should be a dictionary where the keys are the unique values of the `hue` column, and the values are color codes (in hex, RGB, etc.).

    - hue (str): The categorical variable for which the percentage distribution will be displayed.
        This is the column in the `df` that you want to analyze (e.g., 'Exited').
    
    The function calculates the percentage distribution of the `hue` column, creates a bar chart
    using seaborn, and applies a custom palette. Annotations are added above each bar to display
    the percentage value.
    """

    percentage_exited = df[hue].value_counts(normalize=True) * 100
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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}%', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, -10), 
                    textcoords='offset points',
                    fontsize=11, color='white', fontweight='bold')


    plt.tight_layout()
    
    
def plot_custom_catplot(df, custom_palette, items, plot_type, hue):
    """
    Plots stacked bar charts (or other specified types) for categorical variables.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data.
    - custom_palette (dict or list): The color palette for the plot.
    - items (list): List of categorical variables to plot.
    - plot_type (str): Type of plot ('bar', 'point', etc.). Default is 'bar'.
    """
    for item in items:

        grouped = df.groupby([item, hue]).size().reset_index(name='Count')

        total_count = grouped['Count'].sum()
        grouped['percentage'] = (grouped['Count'] / total_count) * 100


        # Define the size based on the number of unique categories in the item column
        if df[item].nunique() <= 5:
            height = 4  # Smaller height if less than 5 categories
            aspect = 1.5  # Adjust aspect ratio
        else:
            height = 6  # Default height for more categories
            aspect = 2   # Default aspect ratio
            
        g = sns.catplot(
            data=grouped, kind=plot_type,
            x=item, y="Count", hue=hue,
            palette=custom_palette, height=height, aspect=aspect,
            errorbar=None
        )


        plt.title(f'% {hue} or Not by {item}', fontsize=16, weight='bold', pad=20)


        for p in g.ax.patches:
            height = p.get_height()
            percentage = (height / total_count) * 100
            if percentage != 0:
                g.ax.annotate(f'{percentage:.1f}%', 
                            (p.get_x() + p.get_width() / 2., height),
                                xytext=(0, -10),  
                                textcoords="offset points",
                                ha='center', va='center',
                                fontsize=11, color='white', fontweight='bold')
            
            

        g.ax.yaxis.set_visible(False)  
        g.ax.spines['top'].set_visible(False)  
        g.ax.spines['right'].set_visible(False)  
        g.ax.spines['left'].set_visible(False)  
        g.ax.spines['bottom'].set_visible(False) 
        g.ax.grid(False)  
        
        g.ax.set_xlabel('')
        

        g.ax.invert_xaxis()
        g._legend.set_title(hue)  
        g._legend.set_bbox_to_anchor((1.15, .8))  


        plt.tight_layout()


def plot_custom_scatterplot(df, x, y, hue, custom_palette, title_fontsize=16, label_fontsize=14):
    """
    Plots a customized scatter plot with specified x and y axes, hue for color coding, and custom palette.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - x (str): Column name for x-axis.
    - y (str): Column name for y-axis.
    - hue (str): Column name for hue (color coding).
    - custom_palette (dict or list): Color palette for the plot.
    - title_fontsize (int): Font size for the title. Default is 16.
    - label_fontsize (int): Font size for x and y labels. Default is 14.
    - show_grid (bool): Whether to display grid lines. Default is False.
    """
    plt.figure(figsize=(16, 6))

    # Create the scatter plot
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=custom_palette, alpha=0.6)

    # Add title and labels
    plt.title(f'{x} vs {y}', fontsize=title_fontsize, weight='bold')
    plt.xlabel(x, fontsize=label_fontsize)
    plt.ylabel(y, fontsize=label_fontsize)

    # Access the current Axes instance
    ax = plt.gca()

    # Customize the plot aesthetics
    ax.spines['top'].set_visible(False)  # Hide the top spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    ax.spines['left'].set_visible(False)  # Hide the left spine
    ax.spines['bottom'].set_visible(False)  # Optionally show the bottom spine
    
    # Customize the legend
    legend = ax.get_legend()
    if legend:
        legend.set_title(hue)  # Set the legend title
        legend.set_bbox_to_anchor((1.15, 0.8))  # Adjust legend position

    # plt.tight_layout()  # Adjust layout to prevent overlap
    
    

def data_visualizations(df, features, color='#c3e88d', hue=None, boxplot=False, histogram=None,
                        barplot=None, custom_palette=None, kde=False, figsize=(24, 12)):
    """
    Plots boxplots and/or histplots for specified items in the DataFrame, optionally separated by hue, in a single figure.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - items (list): List of columns to plot boxplots or histplots for.
    - color (str, optional): Color for the boxplot if hue is not used.
    - hue (str, optional): Column name to separate the boxplots or histplots by hue.
    - boxplot (bool, optional): If True, plots boxplots.
    - histplot (bool, optional): If True, plots histograms.
    - custom_palette (dict or list, optional): Color palette for the plot. Used if hue is provided.
    - figsize (tuple, optional): Figure size.
    """
    num_features = len(features)
    if num_features == 0:
        print("No items to plot.")
        return
    # if barplot:
    #     rows = (num_features + 2) // 2
    #     cols = min(2, num_features)
    # else:
    rows = (num_features + 2) // 3
    cols = min(3, num_features)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
         
        if barplot:
            grouped = df.groupby([feature, hue]).size().reset_index(name='count')

            total_count = grouped['count'].sum()
            grouped['percentage'] = (grouped['count'] / total_count) * 100
            
            num_categories = df[feature].nunique()
            
            # Ajustar dinamicamente a largura das barras
            if num_categories <= 5:
                width = 0.8
            else:
                width = 0.6  # Reduzir a largura das barras para muitas categorias
            
            sns.barplot(data=grouped, x='count', y=feature, palette=custom_palette, hue=hue, ax=ax, width=width, orient='h')
            
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            ax.yaxis.set_visible(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)

            # Ajustar rótulos no eixo y se houver muitas categorias
            if num_categories > 5:
                ax.tick_params(axis='y', rotation=45)  # Rotacionar os rótulos do eixo y se houver muitas categorias

            # Anotações de porcentagem
            for p in ax.patches:
                width = p.get_width()
                percentage = (width / total_count) * 100
                if percentage != 0:
                    ax.annotate(f'{percentage:.1f}%', 
                                (width, p.get_y() + p.get_height() / 2),
                                xytext=(5, 0),  # Ajustar posição do texto
                                textcoords="offset points",
                                ha='left', va='center',
                                fontsize=11, color='black', fontweight='bold')
    
        if histogram:
            sns.histplot(data=df, x=feature, hue=hue, palette=custom_palette, kde=kde, ax=ax, stat='proportion')
            ax.set_xlabel('')
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            ax.yaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)
            
        if boxplot:
            if hue:
                sns.boxplot(data=df, x=feature, y=hue, hue=hue, orient='h', palette=custom_palette, ax=ax)

            else:
                sns.boxplot(data=df, x=feature, color=color, orient='h', ax=ax)
            ax.set_xlabel('')
        
        
            ax.set_title(f'{feature}', fontsize=16, weight='bold')
            ax.yaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False)

    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()