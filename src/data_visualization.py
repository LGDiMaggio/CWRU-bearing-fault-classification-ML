import seaborn as sns
import matplotlib.pyplot as plt

# Set overall aesthetics for all plots
sns.set_theme(style="white")  # Set the style of the plots
plt.rcParams.update({
    'font.size': 18,       # Set the base font size
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.titlesize': 20,  # Set the font size for titles
    'axes.labelsize': 18,   # Set the font size for labels
})

def plot_histogram(dataset, feature, hue='Specific Label', bins=30):
    """Plot a histogram with frequency distribution for a specified feature."""
    plt.figure(figsize=(12, 6))  # Adjusted figure size for visibility
    ax = sns.histplot(data=dataset, x=feature, hue=hue, bins=bins, stat='count', common_norm=False, kde=False)
    #plt.title(f'{feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.tight_layout()


def plot_violin_plot(dataset, feature, hue='Health State', split_by='Signal Type'):
    """Plot a violin plot for a specified feature based on a categorical variable defined in 'hue'."""
    plt.figure(figsize=(12, 6))  # Adjusted figure size for better visibility
    sns.violinplot(x=hue, y=feature, data=dataset, palette='muted', split=False, hue=split_by, bw_method="silverman")
    #plt.title(f'{feature} by {hue}')
    plt.xlabel(hue)
    plt.ylabel(feature)
    plt.tight_layout()

