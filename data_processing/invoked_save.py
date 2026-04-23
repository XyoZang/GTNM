import pickle
import os
import sys
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
    data_path = sys.argv[1] if len(sys.argv) > 1 else './predata'
    prefix = sys.argv[2] if len(sys.argv) > 2 else 'train_subword'
    extract_invoked_data(data_path, prefix)