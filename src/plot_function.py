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

    # Calcular as porcentagens de churn e não-churn
    percentage_exited = df[hue].value_counts(normalize=True) * 100
    percentage_df = percentage_exited.reset_index()
    percentage_df.columns = [hue, 'Percentage']

    # Criando a figura e o gráfico de barras com seaborn
    plt.figure(figsize=(4, 3))
    
    # Aplicando a paleta personalizada de acordo com os valores de 'hue'
    ax = sns.barplot(data=percentage_df, x=hue, y='Percentage', hue=hue, palette=custom_palette)

    # Adicionando título e ajustando o layout
    ax.set_title(f'{percentage_exited.idxmin()} rate is about {percentage_exited.min():.2f}%', 
                 fontweight='bold', fontsize=13, pad=15, loc='center')
    ax.set_xlabel('')
    
    # Inverter o eixo x para uma melhor visualização
    ax.invert_xaxis()

    # Remover ticks e labels dos eixos
    ax.tick_params(axis='both', which='both', length=0)
    
    # Esconder eixo y e remover espinhas (borders)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    
    # Adicionando as anotações de porcentagem acima das barras
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}%', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, -10), 
                    textcoords='offset points',
                    fontsize=11, color='white', fontweight='bold')

    # Ajustando o layout para evitar cortes
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

   
        g = sns.catplot(
            data=grouped, kind=plot_type,
            x=item, y="Count", hue=hue,
            palette=custom_palette, height=6, aspect=2,
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