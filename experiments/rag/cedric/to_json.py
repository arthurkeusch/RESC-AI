import json
import os
import pickle
import sys


if  __name__ == "__main__":
    in_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), sys.argv[1])
    with open(in_path, "rb") as f:
        data = pickle.load(f)

    out_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), sys.argv[2])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4,
                  default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)