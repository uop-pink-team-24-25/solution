import os
import sys

import multiprocessing

from time import sleep

from src.model_runner import ai_model

def prepare_model(path, show):
    selected_model = ai_model(r"C:\Users\jorda\Documents\GitHub\solution\model_tests\yolov5-deepsort\config.yml", show);
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

    process1 = multiprocessing.Process(target=prepare_model, args=(r"C:\Users\jorda\Documents\GitHub\solution\model_tests\yolov5-deepsort\config.yml", show,))

    process2 = multiprocessing.Process(target=prepare_model, args=(r"C:\Users\jorda\Documents\GitHub\solution\model_tests\yolov5-deepsort\config.yml", show,))

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

    import csv
    import json
    import numpy as np
    from src.model_runner import ai_model

    def convert_to_builtin_type(obj):
        if isinstance(obj, dict):
            return {k: convert_to_builtin_type(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [convert_to_builtin_type(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


    model = ai_model("./config.yml", show=False)
    model.run_model()

    # Serialize objects_no_longer_in_scene into string representations
    objects = model.get_objects_no_longer_in_scene()

    with open("output.csv", "w", newline="") as csvfile:
        fieldnames = ["track_id", "start_frame", "end_frame", "vehicle_type", "vehicle_colour", "track_history"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in model.completed_vehicle_data:
            track_id = row["track_id"]
            # Serialize the track history if available, converting numpy types first
            track_history = objects.get(track_id, [])

            print(f"Track history for {track_id} before cleaning:", track_history)

            track_history_clean = convert_to_builtin_type(track_history)
            print(f"Track history for {track_id} after cleaning:", track_history_clean)
            track_history_str = json.dumps(track_history_clean)

            row["track_history"] = track_history_str
            writer.writerow(row)
