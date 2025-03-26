
import json

from typing import Self, Dict
from interfaces import Martial

class JsonMartialler(Martial):
    def serialise(self: Self, data: Dict) -> bytes:
        return json.dumps(data).encode()
