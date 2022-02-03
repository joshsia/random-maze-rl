# random-maze-rl

## About
Using reinforcement learning to solve a randomly generated maze. Some examples of mazes are shown below

## Usage

It is recommended to run the scripts in a virtual environment. To get started, create a virtual environment named `randommaze` by running the following command at the command line:

```bash
conda create --name randommaze python=3.8.2 -y
```

Next, activate the virtual environment:

```bash
conda activate randommaze
```

Then, install the dependencies listed below using:

```bash
pip3 install numpy==1.19.3 matplotlib==3.3.0 opencv-python==4.5.1.48
```
To install pytorch, visit the [pytorch website](https://pytorch.org/get-started/previous-versions/) and look for version 1.7.0. Copy the installation command and run it at the command line. For instance, the installation command for Mac is:

```bash
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 -c pytorch -y
```

Finally, run the python script to train the agent using:

```bash
python train_and_test.py
```

## Dependencies
- numpy=1.19.3
- matplotlib=3.3.0
- opencv-python=4.5.1.48
- torch=1.7.0
