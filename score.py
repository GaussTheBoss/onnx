# modelop.schema.0: input_schema.avsc
# modelop.slot.1: in-use
# modelop.recordsets.0: true


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


# modelop.score
def action(data):
    """
    A function to score input data (real-time inferences)

    param: data: input data in CSV format - headers ignored
    return: predictions (as an array of dictionaries)
    """

    # Input data is a DataFrame; exract features into np.array
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
    return out_df.to_dict(orient='records')


# Test Script
if __name__=='__main__':
    begin()
    input_data = pd.read_csv('score_input_data.csv')
    print(action(input_data))