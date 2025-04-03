from abc import ABC, abstractmethod
from typing import Dict

class Input(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def isOpened(self):
        pass


class PreProcess(ABC):
    pass

class Martial(ABC):
    pass

class Send(ABC):
    def __init__(self, martialler: Martial):
        self.martialler = martialler

    @abstractmethod
    def send(self, data: Dict) -> bool:
        pass
