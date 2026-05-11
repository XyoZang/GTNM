import pickle
import os
import argparse
from tqdm import tqdm

def extract_invoked_data(data_path, prefix):
    print("processing {}".format(prefix))
    pro_data = pickle.load(open(os.path.join(data_path, prefix+'_pro.pkl'), "rb"))
    body_data = pickle.load(open(os.path.join(data_path, prefix+'_body.pkl'), "rb"))
    invoked = []
    for i in tqdm(range(len(pro_data))):
        invoked_data = []
        pro_cxt = pro_data[i]
        body_cxt = body_data[i]
        for pro in pro_cxt:
            if pro in body_cxt:
                invoked_data.append(1)
            else:
                invoked_data.append(0)
        invoked.append(invoked_data)
    pickle.dump(invoked, open(os.path.join(data_path, prefix+'_invoked.pkl'), "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract invoked mask for project context.')
    parser.add_argument("--data_path", type=str, default="./",
                        help="dir to save the final data for training and evaluation")
    parser.add_argument("--prefix", type=str, default="train_subword",
                        help="data prefix, for example: train_subword")
    args = parser.parse_args()
    extract_invoked_data(args.data_path, args.prefix)