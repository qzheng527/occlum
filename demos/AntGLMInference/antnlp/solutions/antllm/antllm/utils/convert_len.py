import os
import pickle
from glob import glob
import numpy as np

for fname in glob("corpus/*/*.len.pkl") + glob("corpus/*/*/*.len.pkl"):
    dname = fname[: fname.rfind("/")] + "/prompt.len.pkl"
    if os.path.exists(dname):
        continue
    d = pickle.load(open(fname, "rb"))
    d = np.array(d)
    print(fname, " -> ", dname)
    d = [0] * len(d)
    with open(dname, "wb") as w:
        pickle.dump(d, w)
#  np.save(dname, d)
