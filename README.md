# TweetsClassification

A predictive model to classify tweets.

## Prerequisite
You need at least 3gb of RAM to install spaCy package.
(tested on Ubuntu system)


If you want to add more training datain the corpus folder, please refer to the annotation guide and be sure that the file name started with "groupe".

## Installation

Use the setup file.

```bash
./setup.sh
```

## Usage

For training the model use:

```python
python3 train.py
```

For prediction use:

```python
python3 predict.py file1 file2 ...
```
