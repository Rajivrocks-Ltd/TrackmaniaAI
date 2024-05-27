import wandb
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

import tmrl.config.config_constants as cfg

wandb_run_id = cfg.WANDB_RUN_ID  # change this by a name of your choice for your run
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]  # name of the wandb project in which your run will appear
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

os.environ['WANDB_API_KEY'] = wandb_key  # this line sets your wandb API key as the active key


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


def parse_run_time(run_time_str):
    """
    Parses a run time string in the format '<99h 99m 59s>' and returns the total time in seconds.

    Parameters:
    - run_time_str: str, run time string in the format '<99h 99m 59s>'

    Returns:
    - total_seconds: int, total run time in minutes
    """
    match = re.match(r'(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?', run_time_str)
    if match:
        hours, minutes, seconds = match.groups()
        hours = int(hours) if hours else 0
        minutes = int(minutes) if minutes else 0
        seconds = int(seconds) if seconds else 0
        total_seconds = hours * 60 + minutes + seconds / 60
        return total_seconds
    else:
        raise ValueError(f"Invalid run time format: {run_time_str}")


def process_folder_name(folder_name):
    """
    Processes the folder name to only use the first and last word in the string,
    split by underscores.

    Parameters:
    - folder_name: str, the original folder name

    Returns:
    - processed_name: str, the processed folder name
    """
    parts = folder_name.split("_")
    if len(parts) > 1:
        return f"{parts[0]} {parts[-1]}"
    else:
        return folder_name


def plot_run_times(data_dir):
    """
    Plots a bar graph of the run times for each folder in the data directory.

    Parameters:
    - data_dir: str, the directory containing the subdirectories with 'run_time.txt' files
    """
    run_times = {}

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            run_time_file = os.path.join(folder_path, 'run_time.txt')
            if os.path.exists(run_time_file):
                with open(run_time_file, 'r') as file:
                    run_time_str = file.readline().strip()
                    try:
                        total_minutes = parse_run_time(run_time_str)
                        run_times[folder] = total_minutes
                    except ValueError as e:
                        print(e)

    # Sort by folder name for consistent bar placement
    sorted_folders = sorted(run_times.keys())
    sorted_times = [run_times[folder] for folder in sorted_folders]
    processed_names = [process_folder_name(folder) for folder in sorted_folders]

    plt.figure(figsize=(12, 6))
    plt.bar(processed_names, sorted_times, color='orange')
    plt.xlabel('Models')
    plt.ylabel('Run Time (minutes)')
    plt.title('Train Time for Each Model in Minutes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if not os.path.exists("plot"):
        os.makedirs("plot")
    plt.savefig(f"{os.getcwd()}/plot/run_times.png")

    plt.show()



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
    # metrics = ["return_train.csv", "loss_actor.csv", "loss_critic.csv"]
    smooth_metrics = ["return_train.csv", "loss_actor.csv", "loss_critic.csv"]

    dir1, dir2 = return_model_dirs("DDPG") # Fetch the directories for the given model name
    for metric_file in metrics: # For all metrics, compare the two models with a plot
        compare_metrics(dir1, dir2, metric_file, smooth_metrics=smooth_metrics, window_size=10, save=True)

    # use this to plot the run times for each folder in the data directory
    plot_run_times("data")
