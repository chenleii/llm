import numpy as np
import json

matrix_random = np.random.rand(2048, 2048).tolist()
ctx = {
    "text":matrix_random,
    "float":[1.0],
    "long":1,
    "urlKeys":[]
}
j = ["CommonQuakeScriptTitle", ctx, "twister"]

with open("/Users/chen/Desktop/aaa.txt","w") as f:
    json.dump(j,f)
