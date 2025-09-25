import time
import torch
import cv2 as cv
from digit import Digit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_model(model, image):
    s = time.time()
    out = model(image)
    pre_lab = torch.argmax(out, dim=1)
    cost_time = round(time.time() - s, 5)
    return cost_time

image = cv.imread("image/sample.png")
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image = 1 - image / 255.
image = cv.resize(image, (8, 8))


image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).contiguous()
origin_model = Digit()
origin_model.load_state_dict(torch.load("model/digit.pth"))
jit_model = torch.jit.load("model/digit.jit")

# init jit
for _ in range(3):
    run_model(origin_model, image)
    run_model(jit_model, image)

test_times = 10

# begin testing
results = pd.DataFrame({
    "type" : ["orgin"] * test_times + ["jit"] * test_times,
    "cost_time" : [run_model(origin_model, image) for _ in range(test_times)] + [run_model(jit_model, image) for _ in range(test_times)]
})

plt.figure(dpi=120)
sns.boxplot(
    x=results["type"],
    y=results["cost_time"]
)
plt.show()