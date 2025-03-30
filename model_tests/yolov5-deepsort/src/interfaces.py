from abc import ABC, abstractmethod
from typing import Dict
from cv2 import VideoCapture, destroyAllWindows

class Input(ABC):
    cap: VideoCapture

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()
        destroyAllWindows()

    def isOpened(self):
        return self.cap.isOpened()


class Martial(ABC):
    @abstractmethod
    def serialise(self, data: Dict) -> bytes:
        pass

class Send(ABC):
    martialler: Martial

    @abstractmethod
    def send(self, data: Dict) -> bool:
        pass
