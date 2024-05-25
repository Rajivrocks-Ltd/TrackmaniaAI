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
    plotter = DataCollection("tmrl", "tmrl")

    data = plotter.collect_data_for_metric(run_id=wandb_run_id)

    # # if the data directory doesn't exist create it and save the data
    # if not os.path.exists("data"):
    #     os.makedirs("data")
    #
    # data is a pandas dataframe, save it as a csv file
    model_dir = "DPPG_CSVs_LIDAR"
    if not os.path.exists(f"data/{model_dir}/full_data"):
        os.makedirs(f"data/{model_dir}/full_data/")

    data.to_csv(f"data/{model_dir}/full_data/{wandb_run_id}x_data.csv")

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
