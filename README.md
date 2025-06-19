# AI image Detection of Live Traffic Feed

![Untitled design](https://github.com/user-attachments/assets/a6932b96-6c6a-4910-9986-7c157a30c661)

## Setup

To setup the prototype, the following dependencies must first be installed:

- [Pipenv]([url](https://pypi.org/project/pipenv/))

Once Pipenv is installed:

1. Clone the 'main' repository to the local device.
2. Navigate to the '...\solution' folder. If on a Linux device, rename 'Pipfile_linux' to 'Pipfile' and replace the default windows pipfile.
3. In '...\solution\model_tests\yolov5-deepsort', create a folder named 'data' and copy in 2 720p video files labelled 'Data.mp4' and 'Data2.mp4'.  (if you have more or less videos, this can be adjusted in the 'user_interface.py' file)
4. Open a terminal in '...\solution' and run the command: ```pipenv install```.
5. Once this has finished, run: ```pipenv shell``` and navigate to the '...\solution\model_tests\yolov5-deepsort\src' directory in the terminal.
6. Finally, run: ```python user_interface.py```
