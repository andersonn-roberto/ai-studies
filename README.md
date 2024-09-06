# AI Studies
Repository of resources to learn about AI

- The fist step to better understand what is and what we can do wit AI is to learn aboud Deep Learning

(Deep Learning: A Crash Course (2018) | SIGGRAPH Courses)[https://www.youtube.com/live/r0Ogt-q956I]


-  If you're thinking hands-on, you might like to start with the `fashion mnist` dataset
The Fashion MNIST dataset is commonly used for training machine learning models, contains 70.000 images in grayscale size 28x28 of fashion products(e.g. t-shirt, trousers, dress, sneaker etc). The training consists of 60.000 imagens and the test set 10.000 imagens.

Python code

```
#packages
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

#import data from train 
training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor()
)

labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",    
}
#plot chart
figure = plt.figure(figsize = (8,8))
cols, rows = 3,3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size =(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

```

Source: https://en.wikipedia.org/wiki/Fashion_MNIST