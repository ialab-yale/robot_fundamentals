# Fundamentals of Robotics and Control

This repository sets you up for the MENG 443/ENAS 773 robotics course assignments. Specifically, this repo has a conda environment and a test script to ensure that you have the necessary coding tools to build, design and control your own or existing robots.

## Conda Install 
To start, install [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your system. We will be creating a virtual environment to ensure that the installed packages do not mess with your system settings.

**Windows Users**: It is recommended that you use WSL2.0 with Ubuntu 20.04 on your system. If you are on Windows 11, the WSL environment will automatically push through and graphics. For those on Windows 10, please see me to get more detailed instructions on how to get set up. 

**Mac Users**: This conda environment was tested on a Arm M1 Mac processor. No issues should be present for intel chipsets. 

**Linux Users**: The conda environment was tested on Ubuntu 20.04.


## Creating the enviroment 
Once you have figure out your setup, install git and clone this repository to any directory of your choosing (WSL folks should clone from the command line in Ubuntu subsystem). Once downloaded, work your way into the directory through the commands line and create the conda environment. 

```conda env create -f meng443_conda_env.yml```

This may require sudo access and take a bit of time. All of the necessary software will be installed to your enviroment. In addition, this will install jupyter. Ensure that the environment has installed correctly and activate the environment with

```conda activate meng443```

Ensure that you have indeed activated the enviroment with 

```conda info```

and search for `active environment : meng443`

To deactivate the environment use the following command 

```conda deactivate```

which will return you to the base conda environment if you have set it up that way. 

## Testing out Jupyter
Please access jupyter using the following command

`jupyter-lab`

This will create a local server which has your running jupyter notebook (if the window does not appear, look for the 127.0.0.1:888...... line of text after you ran the command, this will be the address you can put in your browser). Make sure that you have your `meng443` environment activated. Once jupyter is accessed, please ensure that you are using the Python [conda env:meng443] by creating a dummty notebook using that environment. If this is all successful then congratulations! 

## Testing simulator
Open up the `getUrRobot.ipynb` notebook provided by the repo and follow along through the notebook. 