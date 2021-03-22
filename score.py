# modelop.slot.0: in-use
# modelop.slot.1: in-use


import onnxruntime as rt
import numpy as np
import pandas as pd


# modelop.init
def begin():
    """
    A function to load trained model globally for inference in action function
    """
    global sess, input_name, label_name
    
    # Start an Inference Session by loading trained model in .onnx format
    sess = rt.InferenceSession("rf_iris.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pass


# modelop.score
def action(data):
    """
    A function to score input data (real-time inferences)

    param: data: input data in CSV format - headers ignored
    return: predictions (as an array of dictionaries)
    """

    # load data as a DataFrame, exract features into np.array
    data = pd.DataFrame(data, index=[0])
    data = np.array(data)

    # Compute predictions using trained Classifier
    pred_onnx = sess.run(
        [label_name], 
        {input_name: data.astype(np.float32)}
    )[0]

    # Assign predictions to a DataFrame with id column
    out_df = pd.DataFrame(columns=['id','prediction'])
    out_df['id'] = range(len(data))
    out_df['prediction'] = pred_onnx
    
    # Output results as a JSON-serializable object (array of records/dicts)
    yield out_df.to_dict(orient='records')