import os
import sys

from time import sleep

from src.model_runner import ai_model



if __name__ == "__main__":
    print("wow")
    # Add the src directory to the module search path
    sys.path.append(os.path.abspath('../src')) #should point here

    selected_model = ai_model("./config.yml")

    print("WOWEE IT HAS WORKED YAHOOO")

    #sleep(10000)

    selected_model.run_model()
