import sys
import data.preprocess as pp
import model.predict as pred

def main(mode):
    if mode == "preprocess":
        pp.preprocess_data()
    elif mode == "predict":
        pred.predict()
    elif mode == "all":
        pp.preprocess_data()
        pred.predict()
    else:
        print("Invalid mode. Use 'preprocess', 'predict', or 'all'.")

if __name__ == "__main__":
    mode = sys.argv[1].lower()
    main(mode)
