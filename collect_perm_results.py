import glob
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filepath')
opts = parser.parse_args()
directory = opts.filepath

pat = directory + '/*/logfile.log'
dump = directory + '/final.log'
ls = glob.glob(pat)

t = []
for name in ls:
    print(name.split('/')[-2], end = ' : ')
    with open(name) as f:
        log_lines = f.read().splitlines()
    log_lines = list(filter(lambda x: 'BEST TEST' in x, log_lines))
    test_results = list(map(lambda x: float(x.split()[-1]), log_lines))
    if len(test_results) == 5:
        t.append(test_results)
        if 'perm_0' in name:
            print( np.array(test_results))
        else:
            print(test_results)
a = np.mean(t, axis = 0)
print('TB')
print(a)
with open(dump, 'w') as f:
    json.dump(list(a), f, ensure_ascii=False)