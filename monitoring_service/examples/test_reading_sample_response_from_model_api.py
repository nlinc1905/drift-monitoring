import json
import datetime
import pandas as pd
import numpy as np


"""
Use this file to test what would be coming out of the model_api and being 
passed to monitoring_api.  This is used to test examples/example_run_request.py
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


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


with open(json_fp, 'r') as j:
    # copy the reading process from example_run_request.py
    d = json.loads(j.read())
    print("Prod Data:\n", d, "\n")  # this is what the monitoring_api will get
