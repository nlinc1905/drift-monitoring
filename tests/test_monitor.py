import unittest
import json
import pandas as pd
import numpy as np

from monitoring_service.monitor import Monitor


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


class TestMonitor(unittest.TestCase):

    def test_iterate_false(self):
        json_fp = "data/sample_response_from_model_api.json"
        with open(json_fp, 'r') as j:
            # copy the reading process from example_run_request.py
            d = json.loads(j.read())
            model_api_response_df = pd.read_json(d, orient="index")
        data = pd.concat([model_api_response_df]*100)
        m = Monitor(client_id="1", reference_data=None)
        self.assertFalse(m.iterate(new_rows=data.iloc[:3]))
        self.assertTrue(m.iterate(new_rows=data.iloc[:40]))
