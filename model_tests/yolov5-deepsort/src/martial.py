
import json

from typing import Self, Dict
from interfaces import Martial

class JsonMartialler(Martial):
    def serialise(self: Self, data: Dict) -> bytes:
        return json.dumps(data).encode()

import json
import numpy as np

def convert_to_builtin_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_to_builtin_type(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating, np.ndarray)):
        return obj.item()
    else:
        return obj

class TypeMartialler(Martial):
    def serialise(self: Self, data: Dict):
        (row, objects) = data
        track_id = row["track_id"]
        track_history = objects.get(track_id, [])
        # print(f"Track history for {track_id} before cleaning:", track_history)
        track_history_clean = convert_to_builtin_type(track_history)
        # get a string repr for this.
        row["track_history"] = json.dumps(track_history_clean)

        return row
