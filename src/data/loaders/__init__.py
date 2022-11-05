import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    """
    Stores as JSON a numpy.ndarray or any nested-list composition.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
