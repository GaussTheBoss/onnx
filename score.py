# modelop.slot.0: in-use
# modelop.slot.1: in-use


import onnxruntime as rt
import numpy as np
import pandas as pd


# modelop.init
def begin():

    global sess, input_name, label_name
    
    sess = rt.InferenceSession("rf_iris.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pass

# modelop.score
def action(data):
    
    print(type(data), flush=True)
    
    data = pd.DataFrame(data, index=[0])

    print(data, flush=True)

    data = np.array(data)

    print(data, flush=True)

    pred_onnx = sess.run(
        [label_name], 
        {input_name: data.astype(np.float32)}
    )[0]

    out_df = pd.DataFrame(columns=['id','prediction'])
    out_df['id']=range(len(data))
    out_df['prediction'] = pred_onnx
    

    yield out_df.to_dict(orient='records')