# onnx

This repo serves as an example of how to train and save a model in `ONNX` format (see attached notebook), so it can be loaded at a later time to produce new predictions.

## Running Locally

To run this model locally, create a new Python 3.9.8 virtual environment (such as with `pyenv`). Then, use the following command to update `pip` and `setuptools`:

```
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools
```

And install the required libraries:

```
python3 -m pip install -r requirements.txt
```

The main source code is contained in `score.py`. To compute and write predictions given a file input, run

```
python3 score.py ./data/input_data.csv ./data/output_data.csv
```

## Assets

Model was trained on the `iris` dataset.
 - `trainer.ipynb` is the training notebook.
 - `rf_iris.onnx` is the trained model artifact.
 - `./data/input_data.csv` is a sample file to use for scoring.
