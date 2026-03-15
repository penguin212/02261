import os

for file in os.listdir("data/test"):
    if "dapi" in file:
        os.rename("data/test/" + file, "data/test/" + file.replace("dapi", "DAPI"))
    if "trans" in file:
        os.rename("data/test/" +file, "data/test/" + file.replace("trans", "TRANS"))
    if "T3" in file:
        os.rename("data/test/" + file, "data/test/" + "T0")

for file in os.listdir("data/train"):
    if "dapi" in file:
        os.rename("data/train/" + file, "data/train/" + file.replace("dapi", "DAPI"))
    if "trans" in file:
        os.rename("data/train/" +file, "data/train/" + file.replace("trans", "TRANS"))
    if "T3" in file:
        os.rename("data/train/" + file, "data/train/" + "T0")