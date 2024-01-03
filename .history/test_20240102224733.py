import argparse
from tools import *
from network.phenonet import PhenoNet

def main(phenovit_weight_path, lstm_weight_path, test_dataset_path):
    try:
        check_path(phenovit_weight_path, lstm_weight_path, test_dataset_path)
    except Exception as e:
        print(e)
        return
    model = PhenoNet("cpu", test_dataset_path, phenovit_weight_path, lstm_weight_path)
    print(model._print_architecture())
    # predict_res = model._predict()
    # pheno_phase = read_stage(str(predict_res))

    # print("Phenophase: ", pheno_phase)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phenovit_weight_path", type=str, default="./weights/phenovit_new.pth", help="Path to Phenovit weight file")
    parser.add_argument("--lstm_weight_path", type=str, default="./weights/lstm.pth", help="Path to LSTM weight file")
    parser.add_argument("--test_dataset_path", type=str, default="D:/UserData/Desktop/1/", help="Path to test dataset")
    args = parser.parse_args()

    main(args.phenovit_weight_path, args.lstm_weight_path, args.test_dataset_path)
