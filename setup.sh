#! /bin/bash

# update system
sudo apt -y update && sudo apt -y upgrade

# install python and pip
sudo apt -y install python3
sudo apt -y install python3-pip

# install libraries needed for the classifier
sudo pip3 install -r requirements.txt
sudo python3 -m spacy download fr