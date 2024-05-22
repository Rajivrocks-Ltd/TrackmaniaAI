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


# Define a class that interfaces with Weights & Biases to plot data for specifc runs and metrics given a project name
# and entity name
class Plotting:
    def __init__(self, project_name, entity_name):
        self.project_name = project_name
        self.entity_name = entity_name

    def collect_data_for_metric(self, run_id):
        api = wandb.Api()
        run = api.run(f"{self.entity_name}/{self.project_name}/{run_id}")
        data = run.history()
        return data

    # plot a specific metric from multiple data files, which are .csv files
    def plot_metric(self, metric_name, data_files):
        fig, ax = plt.subplots()
        for data_file in data_files:
            _data = pd.read_csv(data_file)
            ax.plot(_data[metric_name], label=data_file)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name)
        ax.legend()
        plt.title(f"Comparing {metric_name} for all trained models")
        plt.show()


if __name__ == "__main__":
    plotter = Plotting("tmrl", "tmrl")

    data = plotter.collect_data_for_metric("DDPG_logging_test")

    # if the data directory doesn't exist create it and save the data
    if not os.path.exists("data"):
        os.makedirs("data")

    # data is a pandas dataframe, save it as a csv file
    data.to_csv(f"data/{wandb_run_id}_data.csv")

    # get all files from the data directory and make a list, add the data directory to the list
    files = [f"data/{f}" for f in os.listdir("data")]

    # plot the metric "loss_critic" from the data file, just an example for now. But this is a rough framework on how
    # we can plot metrics from multiple runs/models
    plotter.plot_metric("loss_critic", files)
