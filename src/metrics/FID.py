from cleanfid import fid
import argparse


def calculate_fid(dirA, dirB):
    score = fid.compute_fid(dirA, dirB, verbose=False)

    return score

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument( "--dirGT", help="directory containing ground truth images, example: celeba_train/", required = True)
    argParser.add_argument( "--dirEI", help="directory containing enhanced images, example: enhanced/", required = True)

    args = argParser.parse_args()

#Directory of ground truth images 
    dirGT = args.dirGT
    dirEI = args.dirEI
    print(calculate_fid(dirGT, dirEI))
