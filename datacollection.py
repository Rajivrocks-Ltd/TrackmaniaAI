import wandb
import matplotlib.pyplot as plt
import os
import pandas as pd

import tmrl.config.config_constants as cfg

wandb_run_id = cfg.WANDB_RUN_ID  # change this by a name of your choice for your run
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]  # name of the wandb project in which your run will appear
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

os.environ['WANDB_API_KEY'] = wandb_key  # this line sets your wandb API key as the active key


def plot_training_metric(data_dir, metric, specific_model=None):
    """
    Plots the specified training metric for models in the given data directory.

    Parameters:
    - data_dir: str, the directory containing model subdirectories with CSV files
    - metric: str, the metric to plot ('return_train', 'loss_actor', or 'loss_critic')
    - specific_model: str, optional, the specific model directory to plot the metric for
    """
    if specific_model:
        model_dirs = [specific_model] if os.path.isdir(os.path.join(data_dir, specific_model)) else []
    else:
        model_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not model_dirs:
        print("No valid model directories found.")
        return

    plt.figure(figsize=(10, 6))

    for model in model_dirs:
        metric_file = os.path.join(data_dir, model, f"{metric}.csv")

        if os.path.exists(metric_file):
            # Read the CSV file for the given metric
            df = pd.read_csv(metric_file)

            # Assume the first column is the iteration or time step, and the second column is the metric value
            plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=model)
        else:
            print(f"Warning: {metric_file} does not exist for model {model}.")

    plt.xlabel('Iteration/Steps')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f"{metric.replace('_', ' ').title()} over Training")
    plt.legend()
    plt.grid(True)
    plt.show()


def smooth_curve(values, window_size):
    """
    Smooths the curve using a simple moving average.

    Parameters:
    - values: list or array of values to smooth
    - window_size: int, the size of the window to use for smoothing

    Returns:
    - smoothed_values: list of smoothed values
    """
    smoothed_values = []
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        smoothed_values.append(sum(values[start:end]) / (end - start))
    return smoothed_values


def compare_metrics(dir1, dir2, metric_file, smooth_metrics=None, window_size=5, save=False):
    """
    Compares the specified metric between two directories by plotting them on the same graph.

    Parameters:
    - dir1: str, the first directory containing the metric file
    - dir2: str, the second directory containing the metric file
    - metric_file: str, the name of the metric file to compare (e.g., 'return_train.csv')
    - smooth_metrics: list of str, metrics to apply smoothing to
    - window_size: int, the window size for smoothing
    - save: bool, if True, saves the plot to a directory called \plot\<model_name>
    """
    metric_path1 = os.path.join(dir1, metric_file)
    metric_path2 = os.path.join(dir2, metric_file)

    if not os.path.exists(metric_path1):
        print(f"Error: {metric_path1} does not exist.")
        return
    if not os.path.exists(metric_path2):
        print(f"Error: {metric_path2} does not exist.")
        return

    # Read the CSV files for the given metric
    df1 = pd.read_csv(metric_path1)
    df2 = pd.read_csv(metric_path2)

    plt.figure(figsize=(10, 6))

    # Extract base names of directories to use in the legend
    label1 = os.path.basename(os.path.normpath(dir1)).split("_")
    label2 = os.path.basename(os.path.normpath(dir2)).split("_")
    model_name = label1[0]
    label1 = label1[0] + " " + label1[-1]
    label2 = label2[0] + " " + label2[-1]

    # Assume the first column is the iteration or time step, and the second column is the metric value
    x1 = df1.iloc[:, 0]
    y1 = df1.iloc[:, 1]
    x2 = df2.iloc[:, 0]
    y2 = df2.iloc[:, 1]

    # Apply smoothing if specified
    if smooth_metrics and metric_file in smooth_metrics:
        y1 = smooth_curve(y1, window_size)
        y2 = smooth_curve(y2, window_size)

    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)

    plt.xlabel('Iteration')
    plt.ylabel(metric_file.replace('_', ' ').title().replace('.csv', ''))
    plt.title(f"Comparison of {metric_file.replace('_', ' ').title().replace('.csv', '')}")
    plt.legend()
    plt.grid(True)

    # Save the plot if save is True
    if save:
        plot_dir = os.path.join('plot', model_name)
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{metric_file.replace('.csv', '')}_comparison.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    plt.show()


def return_model_dirs(model_name: str) -> tuple:
    """
    Function that selects the two corresponding "pixels" and "LIDAR" directories for a given model name.

    :param model_name: model name to search for corresponding pixel and LIDAR directories
    :return: tuple of directories (dir1, dir2)
    """
    dir1 = None
    dir2 = None

    for model in os.listdir("data"):
        folder = model.split("_")
        if folder[0] == model_name:
            if model.endswith("Pixels"):
                dir1 = f"data/{model}/"
            elif model.endswith("LIDAR"):
                dir2 = f"data/{model}/"

    if dir1 and dir2:
        return dir1, dir2
    else:
        raise ValueError("Could not find corresponding 'pixels' and 'LIDAR' directories for the given model name.")


# Define a class that interfaces with Weights & Biases to plot data for specifc runs and metrics given a project name
# and entity name
class DataCollection:
    def __init__(self, project_name, entity_name):
        self.project_name = project_name
        self.entity_name = entity_name

    def collect_data_for_metric(self, run_id):
        api = wandb.Api()
        run = api.run(f"{self.entity_name}/{self.project_name}/{run_id}")
        _data = run.history()
        return _data


if __name__ == "__main__":
    # uncomment if you want to fetch all data from W&B directly (sometimes it fetches partial data and you need to get
    # the data from the W&B website directly)
    # plotter = DataCollection("tmrl", "tmrl")
    #
    # data = plotter.collect_data_for_metric(run_id=wandb_run_id)

    # # if the data directory doesn't exist create it and save the data
    # if not os.path.exists("data"):
    #     os.makedirs("data")
    #
    # # data is a pandas dataframe, save it as a csv file
    # model_dir = "PPO_CSVs_LIDAR"
    # if not os.path.exists(f"data/{model_dir}/full_data"):
    #     os.makedirs(f"data/{model_dir}/full_data/")
    #
    # data.to_csv(f"data/{model_dir}/full_data/{wandb_run_id}x_data.csv")

    # # get all files from the data directory and make a list, add the data directory to the list
    # files = [f"data/{f}" for f in os.listdir("data")]
    #
    # # plot the metric "loss_critic" from the data file, just an example for now. But this is a rough framework on how
    # # we can plot metrics from multiple runs/models
    # plot_metric("return_train", files)

    # use this to plot the data from the csv file using the plot_training_metric function
    # data_directory = "data"  # Change this to your data directory path
    # metric_to_plot = "loss_actor"  # Change this to the metric you want to plot
    # specific_model = None  # Change this to a specific model directory name, or leave as None for all models
    # plot_training_metric(data_directory, metric_to_plot, specific_model)

    # use this to compare the metrics from two different directories using the compare_metrics function
    metrics = ["return_train.csv", "loss_actor.csv", "loss_critic.csv"]
    smooth_metrics = ["return_train.csv", "loss_actor.csv", "loss_critic.csv"]

    dir1, dir2 = return_model_dirs("DDPG") # Fetch the directories for the given model name
    for metric_file in metrics: # For all metrics, compare the two models with a plot
        compare_metrics(dir1, dir2, metric_file, smooth_metrics=smooth_metrics, window_size=10, save=True)
