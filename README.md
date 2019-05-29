# TweetsClassification

A predictive model to classify tweets.

If you want to add more training data in the corpus folder, please refer to the annotation guide and be sure that the file name starts with "groupe".

## Installation

Use the setup file and decompress the rf_classifier.zip

```bash
./setup.sh
unzip rf_classifier.zip
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
