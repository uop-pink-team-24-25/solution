from abc import ABC, abstractmethod
from typing import Dict, Any
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
    def serialise(self, data: Any) -> Any:
        pass

class Send(ABC):
    martialler: Martial

    @abstractmethod
    def send(self, data: Any) -> bool:
        pass
