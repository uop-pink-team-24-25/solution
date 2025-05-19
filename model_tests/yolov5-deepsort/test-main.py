import os
import sys

import multiprocessing

from time import sleep

from src.model_runner import ai_model

def prepare_model(path, show):
    selected_model = ai_model(path, show);
    selected_model.run_model();

def test():

    multiprocessing.set_start_method('spawn')

    show = False

    if("-show" in sys.argv):
        show = True

#    config_path = sys.argv[1]

#    if(config_path[len(config_path) - 4:len(config_path)] != ".yml"):
#        print("usage: config_path camera_counts (-show)")
#        print(config_path[len(config_path) - 4:len(config_path)])
#        quit()




    print("show = " + str(show))
    print("wow")
    # Add the src directory to the module search path
    sys.path.append(os.path.abspath('../src')) #should point here

    process1 = multiprocessing.Process(target=prepare_model, args=("./config.yml", show,))

    process2 = multiprocessing.Process(target=prepare_model, args=("./config2.yml", show,))

    process1.start();

    print("PROCESS 1 STARTED\n\n\n\n\n\n")

    sleep(20)

    process2.start();

    print("PROCESS 2 STARTED\n\n\n\n\n\n")

    process1.join();

    process2.join();

    print("WOWEE IT HAS WORKED YAHOOO")

    #sleep(20)

    #selected_model.run_model()

if __name__ == "__main__":

    show = False

    if("-show" in sys.argv):
        show = True

    model = ai_model("./config.yml", show);

    model.run_model();
