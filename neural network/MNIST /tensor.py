import os
import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

# -----------------------------------------
# Load pretrained DenseNet
# -----------------------------------------
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

# Freeze pretrained weights
for param in model.parameters():
    param.requires_grad = False

# -----------------------------------------
# New classifier
# -----------------------------------------
# NOTE: This head outputs 2 classes. If you're doing full ImageNet-1000
# classification, change 500 -> whatever hidden size you want and
# 2 -> 1000. This 2-class setup only makes sense if your ImageFolder
# directory actually has exactly 2 class subfolders.
classifier = nn.Sequential(
    nn.Linear(1024, 500),
    nn.ReLU(),
    nn.Linear(500, 2),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

# -----------------------------------------
# Image transforms
# -----------------------------------------
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------------
# Dataset
# -----------------------------------------
DATA_ROOT = "/Users/hindrajprakashmali/waddle/neural network/cat"

# ImageFolder does NOT support `download=True` — it only works with data
# you already have on disk, laid out as:
#   DATA_ROOT/
#       class_a/
#           img1.jpg
#           img2.jpg
#       class_b/
#           img1.jpg
#           ...
if not os.path.isdir(DATA_ROOT):
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_ROOT}\n"
        "ImageFolder needs an existing directory of class subfolders "
        "(it does not download data for you). Point DATA_ROOT at your "
        "actual dataset location, structured as root/class_name/*.jpg"
    )

trainset = datasets.ImageFolder(
    root=DATA_ROOT,
    transform=transform
)

print(f"Found {len(trainset)} images across {len(trainset.classes)} classes: {trainset.classes}")

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)

# -----------------------------------------
# Loss & Optimizer
# -----------------------------------------
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

# -----------------------------------------
# CPU / GPU Benchmark
# -----------------------------------------
for use_cuda in [False, True]:

    if use_cuda and not torch.cuda.is_available():
        print("CUDA requested but not available — skipping this pass.")
        continue

    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.train()

    start = time.time()
    n_batches = 0

    for ii, (inputs, labels) in enumerate(trainloader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        n_batches += 1

        if ii == 3:
            break

    elapsed = time.time() - start
    print(
        f"Device = {device}; "
        f"Time per batch = {elapsed / n_batches:.3f} seconds"
    )