# GroundingDINO-SAM

GroundingDINO + Segment Anything Model plugin for DeepMake

# Plugin for DeepMake
This repo the GroundingDINO + Segment Anything Model plugin for DeepMake.

This plugin is not meant to be run directly, instead it's built to fit into DeepMake.

Installation
To install, you need to download and install Anaconda

Clone this repo into the "plugin" folder of your DeepMake installation.

Next open an Anaconda prompt and go to the plugin/GroundingDINO-SAM
/ folder.

If you're on Windows, Run conda env create -f environment.yml to create the environment.

If you're on a Mac, instead run `conda env create -f environment_mac.yml

Then 
cd GroundingDINO-SAM
pip install -e .

Finally, run DeepMake as normal. `
