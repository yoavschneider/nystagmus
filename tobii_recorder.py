from __future__ import print_function

import tobii_research as tr
import time, sys, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename")
parser.add_argument("-s", "--seconds", type=int)
args, leftovers = parser.parse_known_args()

def gaze_data_callback(gaze_data):
    global res_file

    # Ignore invalid results
    if gaze_data['left_gaze_point_validity'] == 0 or gaze_data['right_gaze_point_validity'] == 0:
        return

    line = "{};{};{};{};{};{};{}".format(time.time(), gaze_data['left_gaze_point_on_display_area'], gaze_data['right_gaze_point_on_display_area'],
        gaze_data['left_gaze_origin_in_user_coordinate_system'], gaze_data['right_gaze_origin_in_user_coordinate_system'], 
        gaze_data['left_gaze_point_in_user_coordinate_system'], gaze_data['right_gaze_point_in_user_coordinate_system'])

    print(line, file=res_file)

with open(args.filename, "w+") as res_file:    
    found_eyetrackers = tr.find_all_eyetrackers()
    eyetracker = found_eyetrackers[0]
    print("Address: " + eyetracker.address)
    print("Model: " + eyetracker.model)
    print("Serial number: " + eyetracker.serial_number)

    # Coordinates of the display area in user coordinate system
    with open("data/tobii_screen_data.txt", "w+") as f:    
        display_area = eyetracker.get_display_area()
        print(display_area.top_left, file=f)
        print(display_area.top_right, file=f)
        print(display_area.bottom_left, file=f)
        print(display_area.bottom_right, file=f)

    with open("data/tobii_calibration_data", "rb") as f:
        calibration_data = f.read()
        if len(calibration_data) > 0:
            eyetracker.apply_calibration_data(calibration_data)
            print("Subscribing to gaze data for eye tracker with serial number {0}.".format(eyetracker.serial_number))
            eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
            time.sleep(args.seconds)
            eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA)
        else:
            raise IOError


