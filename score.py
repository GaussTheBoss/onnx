from sys import argv

import onnxruntime as rt
import numpy
import pandas


def begin():
    """
    A function to load trained model globally for inference in action function
    """
    global sess, input_name, label_name

    # Start an Inference Session by loading trained model in .onnx format
    sess = rt.InferenceSession("rf_iris.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name


def predict(data: pandas.DataFrame) -> pandas.DataFrame:
    """
    A function to score input data (real-time inferences)

    Args:
        data (pandas.DataFrame): Input data to be scored.

    Returns:
        (pandas.DataFrame): Predictions.
    """

    # Input data is a DataFrame; exract features into np.array
    data = numpy.array(data)

    # Compute predictions using trained Classifier
    pred_onnx = sess.run([label_name], {input_name: data.astype(numpy.float32)})[0]

    # Assign predictions to a DataFrame with id column
    output_df = pandas.DataFrame(columns=["id", "prediction"])
    output_df["id"] = range(len(data))
    output_df["prediction"] = pred_onnx

    return output_df


if __name__ == "__main__":
    # Sample Usage from Command Line:
    # python3 score.py ./data/input_data.csv ./data/output_data.csv

    # Load pretrained model
    begin()

    # Read input file
    input_data = pandas.read_csv(argv[1])  # argv[1] is input file path
    # Predict
    scores = predict(input_data)
    # Write to output file
    scores.to_csv(argv[2], index=False)  # argv[2] is output file path
