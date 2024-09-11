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
    percentage_df.columns = [hue, 'Percentage']


    plt.figure(figsize=(4, 3))
    

    ax = sns.barplot(data=percentage_df, x=hue, y='Percentage', hue=hue, palette=custom_palette)


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
        grouped['Percentage'] = (grouped['Count'] / total_count) * 100


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
        # ax.spines['bottom'].set_visible(False) 
        g.ax.grid(False)  
        

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
    
    
def plot_custom_histograms(df, custom_palette, items, hue_choice):
    """
    Plots histograms for specified items in the DataFrame, separated by hue.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - custom_palette (dict or list): Color palette for the plot.
    - items (list): List of columns to plot histograms for.
    - hue_choice (str): Column name to use for hue (color coding).
    """
    for item in items:
        
        plt.figure(figsize=(15, 6))
        
        # Create the histogram plot
        g = sns.histplot(data=df, x=item, hue=hue_choice, 
                         palette=custom_palette, kde=True)#, kde=True, multiple='stack')

        # Add title and labels
        plt.title(f'Histogram of {item} by {hue_choice}', fontsize=16, weight='bold')
        plt.xlabel(item, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        
        # Set x-axis limits
        plt.xlim([df[item].min(), df[item].max()])
        
        # Access the current Axes instance
        ax = plt.gca()
        ax.yaxis.set_visible(False)  # Hide y-axis
        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['left'].set_visible(False)  # Hide the left spine

        # Customize the legend only if it exists
        if ax.get_legend() is not None:
            legend = ax.get_legend()
            legend.set_title(hue_choice)  # Set the legend title
            legend.set_bbox_to_anchor((1.15, 0.8))  # Adjust legend position

        # plt.tight_layout()  # Adjust layout to prevent overlap
        
        
import matplotlib.pyplot as plt
import seaborn as sns

def plot_custom_boxplot(df, items, color=None, hue=None, custom_palette=None):
    """
    Plots boxplots for specified items in the DataFrame, optionally separated by hue, in a single figure.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - items (list): List of columns to plot boxplots for.
    - hue (str, optional): Column name to separate the boxplots by hue.
    - custom_palette (dict or list, optional): Color palette for the plot. Used if hue is provided.
    """
    num_items = len(items)
    rows = (num_items + 2) // 3  # Ajusta o número de linhas dependendo da quantidade de itens
    cols = min(3, num_items)  # Número de colunas ajusta ao número de itens, máximo de 3 colunas
    
    if hue:
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))  # Cria os subplots
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))  # Cria os subplots
    axes = axes.flatten()  # Flatten para o caso do número de itens ser ímpar

    for i, item in enumerate(items):
        ax = axes[i]  # Seleciona o subplot para o item atual

        if hue:
            sns.boxplot(data=df, x=item, y=hue, hue=hue, orient='h', palette=custom_palette, ax=ax)
            # ax.set_ylabel(hue, fontsize=14)
        else:
            sns.boxplot(data=df, x=item, color=color, orient='h', ax=ax)
            # ax.set_ylabel('')
            
        ax.set_xticks([])
        ax.set_title(f'{item}', fontsize=16, weight='bold')
        ax.yaxis.set_visible(False)  # Esconde o eixo y se não estiver usando hue
        ax.spines['top'].set_visible(False)  # Esconde a borda superior
        ax.spines['right'].set_visible(False)  # Esconde a borda direita
        ax.spines['left'].set_visible(False)  # Esconde a borda esquerda
        ax.spines['bottom'].set_visible(False)  # Esconde a borda inferior
        ax.grid(False)

    # Remove gráficos extras vazios
    for j in range(num_items, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

