#! /bin/bash

# install python and pip
sudo apt -y install python
sudo apt -y install python-pip

# install libraries needed for the classifier
sudo pip install -r requirements.txt
sudo python -m spacy download fr