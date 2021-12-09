import json
import pandas as pd
import numpy as np


"""
Use this file to test what would be coming out of the model_api and being 
passed to monitoring_api
"""


# run this from root dir, not in /examples
json_fp = "data/sample_response_from_model_api.json"


_integer_types = (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
_float_types = (np.float_, np.float16, np.float32, np.float64)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, _integer_types):
            return int(o)
        if isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.void):
            return None
        return json.JSONEncoder.default(self, o)


with open(json_fp, 'r') as j:
    # copy the reading process from example_run_request.py
    d = json.loads(j.read())
    print(d)
    model_api_response_df = pd.read_json(d, orient="index")
    print(model_api_response_df)
    data = model_api_response_df.to_dict(orient="records")
    print(data)
    post_request = json.dumps(data, cls=NumpyEncoder)
    print(post_request) # this is what the monitoring_api will get