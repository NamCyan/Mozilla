import os


src = "/datadrive5/namlh35/Mozilla/results/ace"
for method in os.listdir(src):
    for perm in os.listdir(os.path.join(src, method)):
        if not os.path.isdir(os.path.join(src, method, perm)):
            continue
        for filename in os.listdir(os.path.join(src, method, perm)):
            if filename.startswith("model"):
                os.remove(os.path.join(src, method, perm, filename))
