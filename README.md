# MGAIA Assignment 3 Setup
In this README we will guide you through the setup process of TMRL, how to train and how to perform inference. 
We have provided pre-trained models for all our agents in the `model_weights` folder, but if you want to train the model yourself, follow the steps below. 
Provided in our repository is a custom track that we have created for you to train and perform inference on. The tracks 
is called `tmrl-test-track-1` and can be found in the `train_files` folder in the root directory of the project. 
We have also provided a reward pickle file that is specifically tailored towards our own track, 
so it is important that you have this file in the correct directory. 
The reward file is located in the `reward pickle file` folder in the root directory of the project.

## Table of Contents
1. [General Information](#general-information)
1. [First-time-setup](#first-time-setup)
    1. [TMRL and OpenPlanet Installation](#tmrl-and-openplanet-installation)
    2. [Custom Track and Reward File](#custom-track-and-reward-file)
2. [Training](#training)
    1. [Training setup](#training-setup)
    2. [Training process](#training-process)
3. [Inference](#inference)

## General Information
The project is structured as follows:
- `README.md`: The README file that you are currently reading.
- `requirements.txt`: Contains the required packages to run the project.
- `configs`: Contains the `config.json` files for each implemented algorithm
- `train_files`: Contains the custom track and reward pickle file used for training an agent with TMRL.
- `data`: Contains the results from training the agents.
- `model_weights`: Contains the pre-trained models for all our agents.
- `pipelines`: Contains the code for the agents and the pipeline to train and perform inference.

As mentioned before, we have provided pre-trained models for all our agents in the `model weights` folder, 
but if you want to train the model yourself, follow the steps below.


## First-time setup
Before any work can be done, a few things need to be installed and set up. 

### TMRL and OpenPlanet Installation
1. Unzip the provided zip file, navigate to the root directory of the project and open it in your IDE of choice (PyCharm recommended).
2. Read through the [TMRL instructions][install_md] on installing TMRL and OpenPlanet. Following
these instructions, I recommend running ```pip install pywin32``` in a _venv_ instead of ```conda install pywin32```.
3. Follow the instructions to see if everything is installed correctly by following the provided ['Getting started'][getting_started_md] guide. 
4. We recommend re-installing PyTorch after the previous steps are completed and verified. This is in case you want to train yourself.
Please run the following command for venv users: 

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
``` 

and for conda users: 
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

After these steps are completed, you should have TMRL installed and OpenPlanet working within Trackmania. Also,
optionally, you have re-installed PyTorch.


### Custom Track and Reward File
To make training and inference work with our own track, a few files need to be copied to the correct directories.

1. Open the `train_files` folder in the root directory of the project, containing `reward.pkl` and `tmrl-test-track-1.Map.Gbx`.
2. Copy the `reward.pkl` file and navigate to the `C:\Users\USERNAME\TmrlData\reward` directory.
3. Delete the existing `reward.pkl` file and paste the new one in its place.
4. Copy the `tmrl-test-track-1.Map.Gbx` file and navigate to the `C:\Users\USERNAME\Documents\Trackmania\Maps\My Maps` directory.
5. Paste the copied track in the directory.

Following these steps, you should have the `reward.pkl` file in the correct directory and the custom track in the correct directory.
This `reward.pkl` file is specifically tailored towards our own track, so it will not work for other tracks. More 
information on how to create your own reward function can be found in the `tmrl` repository (see: )


## Training

### Training setup
Now we move on to training the model. We have provided pre-trained models for all our agents in the `model weights` folder, but if you want to train the model yourself, follow these steps:
1. Choose a model you want to train, which is either a Pixel or LIDAR-based agent.
2. Navigate to the `configs` directory, dependent on if you want to train LIDAR or Pixel agents, navigate to the corresponding sub-directory.
3. Copy the `config.json` file from the chosen directory.
4. Navigate to the `C:\Users\USERNAME\TmrlData\config` directory and paste the copied `config.json` file in the directory.
5. Open the `config.json` file and change the `RUN_NAME` to a name of your choosing. Everything else is setup for you.
**Note**: if you do not change the `RUN_NAME` and you have trained before using the same `RUN_NAME` the checkpoint file in `C:\Users\USERNAME\TmrlData\checkpoints` will be used to continue training, so changing the `RUN_NAME` is important if you want to start training from scratch or if you want to train different agents.
6. _Optional_: If you are training on a CPU set the `CUDA_TRAINING` flag to `false` in the `config.json` file
7. _Optional_: If you want the model weights to be saved every `x` steps, change the `SAVE_MODEL_EVERY` flag to the desired amount of steps, with `0` meaning **no saving**.

### Training process
Now that we are set-up for training we can start the training process:
1. Open Trackmania, navigate to `Create` in the main menu, then `Track Editor` -> `Edit Track` -> `My Local tracks` -> `My Maps` and you should see the `tmrl-test-track-1` track[^tp-fn]. 
2. Click on `tmrl-test-track-1` and click the `Green Flag` in the bottom right corner. Now you should see the car on the track. Make sure you followed the `Getting started` guide `step 3` because depending on Pixel or LIDAR-based agent, you need to change the camera view and the ghost needs to be turned off for all agents.
3. Take note of your chosen agent you want to train, either Pixel or LIDAR variant of DDPG, PPO, SAC-DC, SAC-SC.
4. Open three separate terminals (make sure you are in your venv or conda env) and navigate to the `pipelines\<YOUR AGENT YOU WANT TO TRAIN>` in all three terminals.
   1. In the first terminal run the following command: `py pipeline.py --server` or `python pipeline.py --server` (depending on if the first one did not work). You should see three lines in the terminal
   2. In the second terminal run the following command: `py pipeline.py --trainer` or `python pipeline.py --trainer` (depending on if the first one did not work). After a few seconds you should see that an empty replay buffer is initialized
   3. In the third terminal run the following command: `py pipeline.py --worker` or `python pipeline.py --worker` (depending on if the first one did not work) You should here a dinging noise once and now if you move to the Trackmania screen you can see that the car is moving. In the console you should see it's collecting samples etc.
5. Move back to the second terminal with the `trainer` running. After a while it will show a round is completed and it has logged some values for you.
6. From here you can sit back and wait until the training process is finished, which is 10 epochs. You can stop the training process at any time by pressing `CTRL + C` in the terminal with the `trainer` running. The model weights will be saved in the `C:\Users\USERNAME\TmrlData\weights` directory (if set.)


## Inference
Now that we have trained the model, we can perform inference on the model. We have provided a pre-trained models for all our agents in the `model_weights` folder, but if you want to perform inference on the model you trained yourself the process will be comparable:
1. Choose the sensory (LIDAR or pixel data) you want to perform inference on.
2. Navigate to the `configs` directory and to the chosen sensory sub-directory.
3. Now choose if you want to perform inference on either the SAC-SC, SAC-DC, DDPG or PPO agent. Copy the `config.json` file from the chosen directory.
4. Navigate to the `C:\Users\USERNAME\TmrlData\config` directory. and paste the copied `config.json` file in the directory.
5. Navigate to the `model_weights` directory. Choose the agent corresponding to the chosen `config.json` file. Copy the `.tmod` file from the chosen directory.
7. Navigate to the `C:\Users\USERNAME\TmrlData\weights` directory. and paste the copied `.tmod` file in the directory.
8. Open a terminal and navigate to the `pipelines\<YOUR AGENT YOU WANT TO PERFORM INFERENCE ON>` directory.
9. Have Trackmania open and the `tmrl-test-track-1` track loaded in the editor. (As described in ``Training`` step 1,2 and 3)
10. Open a terminal and run the following command: `py pipeline.py --test` or `python pipeline.py --test` (depending on if the first one did not work). The car should start performing actions.


Performing inference on the model you trained yourself:
1. After training has finished you can quit the `server`, `trainer` and `worker` terminals.
2. Use one of the terminals you used before or open a new terminal and navigate to the `pipelines\<YOUR AGENT YOU WANT TO PERFORM INFERENCE ON>` directory. (this has to be the one you finished training
3. Have Trackmania open and the `tmrl-test-track-1` track loaded in the editor. (As described in ``Training`` step 1,2 and 3)
4. In the open terminal run the following command: `py pipeline.py --test` or `python pipeline.py --test` (depending on if the first one did not work) The car should start doing stuff


**IMPORTANT NOTICE**  
Inference on the Pixel-based agents is somewhat broken in our TMRL environment. You can perform inference for one round and than the package throws an error. this might not
happen for you, but if it happens please just restart the inference process by repeating the last command in the terminal.

# Enjoy the ride! 

[//]: # (Footnotes)
[^tp-fn]: If you don't see it, make sure the file in `train_files` has correctly been copied to `C:\Users\USERNAME\Documents\Trackmania\Maps\My Maps`)

[//]: # (Links)
[install_md]: https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md
[getting_started_md]: https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md
[reward_function]: https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md#train-your-own-self-driving-ai