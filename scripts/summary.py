import os
import numpy as np

rs_folder = "/datadrive5/namlh35/Mozilla/results/ace/ExcessMTL_woLMH_woSR_woSAM_woLosstrick_narate-40_cluster-8_gen-10_robust_step_size-0.0001"

results = []

for perm_id in os.listdir(rs_folder):
    if not os.path.isdir(os.path.join(rs_folder, perm_id)):
        continue
    perm_rs = []
    with open(os.path.join(rs_folder, perm_id, "LOSS_LOG.txt"), "r") as f:
        for line in f:
            if line.startswith("BEST"):
                perm_rs.append(100*float(line.split("TEST")[-1].split(":")[-1]))
    assert len(perm_rs) == 5
    results.append(perm_rs)

avg = [float(np.mean([perm[i] for perm in results])) for i in range(5)]

with open(os.path.join(rs_folder, "summary.txt"), "a") as f:
    for rs in results:
        f.write(" ".join([str(x) for x in rs])+"\n")
    f.write("-"*20+"\n")
    f.write(" ".join([str(x) for x in avg]))