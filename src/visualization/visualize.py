"""
Script to create visualizations for analysis and reporting
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_plot_style():
    """
    Set up the plotting style
    """
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)


def save_figure(fig, filename):
    """
    Save a figure to the reports/figures directory

    Args:
        fig: Matplotlib figure object
        filename: Name of the file to save
    """
    project_dir = Path(__file__).resolve().parents[2]
    output_path = project_dir / 'reports' / 'figures' / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")


def main():
    """
    Main function to create visualizations
    """
    setup_plot_style()

    # Example visualization
    # fig, ax = plt.subplots()
    # ax.plot([1, 2, 3], [1, 4, 9])
    # ax.set_title('Example Plot')
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # save_figure(fig, 'example_plot.png')

    print("Visualizations created successfully")


if __name__ == '__main__':
    main()
