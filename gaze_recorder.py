import argparse, os

from nyst import gaze_recorder

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--camera")
	parser.add_argument("-f", "--filename")
	parser.add_argument("-s", "--seconds", type=int)
	parser.add_argument("-width", "-wi", type=float)
	parser.add_argument("-height", "-he", type=float)
	parser.add_argument("-fa", "--factors", help="file name for calibration factors")
	args, leftovers = parser.parse_known_args()

	gaze_recorder.start(args.filename, args.factors, args.camera, args.seconds, args.width, args.height)