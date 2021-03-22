# onnx

This repo serves as an example of how to train and save a model in ONNX format (see attached notebook), so it can be conformed to MOC standards (`score.py`).

MOC model loads .onnx trained model artifact in begin() function, and uses it in scoring function to produce predictions on input data (`input_data.csv`).