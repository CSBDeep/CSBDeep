#! /bin/bash   
set -m

pkill tensorboard
pkill jupyter 

# start jupyter notebook, send to background and wait some seconds
jupyter notebook --port=8888 &
sleep 2

# start tb server
tensorboard --logdir=logs -p 6006 --reload_interval 2 &
sleep 2
chromium-browser http://localhost:6006


# open notebooks
chromium-browser http://localhost:8888/notebooks/training-tribolium.ipynb
chromium-browser http://localhost:8888/notebooks/training-retina.ipynb
chromium-browser http://localhost:8888/notebooks/training-tubulin.ipynb
chromium-browser http://localhost:8888/notebooks/datagen.ipynb


#
fg 
fg

