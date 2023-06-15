from collections import defaultdict as ddict
import pprint

data_path = "vd.txt"

data_dict = dict()
with open(data_path, 'r') as f:
    i = 0
    for row in f.readlines():
        data_dict[f"{i}"] = {f"{j}": float(val) for j, val in enumerate(row.split(' '))}
        i += 1
pprint.pprint(data_dict)
