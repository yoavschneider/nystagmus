import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-dh", "--hdata", help="file name for horizontal calibration data")
parser.add_argument("-dv", "--vdata", help="file name for vertical calibration data")
parser.add_argument("-factors", "--factors", help="file name for calibration factors")
parser.add_argument("-f", "--filename", help="file name for saving plot")
args, leftovers = parser.parse_known_args()

SCREEN_WIDTH_MM = 517 
SCREEN_HEIGHT_MM = 324

horizontal_error_left_eye = []
horizontal_error_right_eye = []
vertical_error_both_eyes = []

with open(args.factors, "r") as factors_data:
    left_eye_factors = [float(val) for val in factors_data.readline().split(";")]
    right_eye_factors = [float(val) for val in factors_data.readline().split(";")]
    vertical_factors = [float(val) for val in factors_data.readline().split(";")]

with open(args.hdata, "r") as calibration_data:
    horizontal_error_left_eye = []
    horizontal_error_right_eye = []

    for line in calibration_data:
        values = line.split(";")
        reference, face, left, right = float(values[0]), float(values[1]),  float(values[2]), float(values[3])
        left_normalized = face * left_eye_factors[0] + left * left_eye_factors[1] + left_eye_factors[2]
        right_normalized = face * right_eye_factors[0] + right * right_eye_factors[1] + right_eye_factors[2]

        horizontal_error_left_eye.append(abs(reference - left_normalized) * SCREEN_WIDTH_MM)
        horizontal_error_right_eye.append(abs(reference - right_normalized) * SCREEN_WIDTH_MM)

with open(args.vdata, "r") as calibration_data:
    vertical_error_both_eyes = []

    for line in calibration_data:
        values = line.split(";")
        reference, face, vertical = float(values[0]), float(values[1]), float(values[2])

        vertical_normalized = face * vertical_factors[0] + vertical * vertical_factors[1] + vertical_factors[2]
        vertical_error_both_eyes.append(abs(reference - vertical_normalized) * SCREEN_HEIGHT_MM)


fig, ax = plt.subplots(figsize=(12,1.1))
ax.set_ylabel('Error in millimeters')
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax.set_title(' ' * 21 + 'N Horizontal = ' + str(len(horizontal_error_left_eye)) + ' ' * 52 + 'N Vertical = ' + str(len(vertical_error_both_eyes)), fontsize=13)
labels = ["Left Eye X", "Right Eye X", "Both Eyes Y"]
data = [horizontal_error_left_eye, horizontal_error_right_eye, vertical_error_both_eyes]
ax.boxplot(data, showfliers=False, labels=labels, showmeans=True, meanline=True)

manager = plt.get_current_fig_manager()
manager.resize(1400, 500)

plt.savefig(args.filename, bbox_inches='tight')

with open(args.filename + '.means.csv', "w+") as f:
	line = "{};{};{}".format('Horizontal Left Eye','Horizontal Right Eye','Vertical Both Eyes')
	print(line, file=f)
	line = "{};{};{}".format(np.around(np.mean(horizontal_error_left_eye), decimals=2), np.around(np.mean(horizontal_error_right_eye), decimals=2),np.around(np.mean(vertical_error_both_eyes),decimals=2))
	print(line, file=f)
