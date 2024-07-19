import torch
import csv
import numpy as np
import matplotlib.pyplot as plt

# import data
def remove_comma(n):
    while "," in n:
        n = n.replace(",", "")

    return float(n)

# import data
y, x1, x2, x3 = [], [], [], []
with open("data.csv", encoding = "utf-8") as f:
    reader = list(csv.reader(f))
    print(reader[0])

    for row in reader[1: ]:
        y.append([remove_comma(row[4])])
        x1.append([remove_comma(row[3])])
        x2.append([remove_comma(row[2])])
        x3.append([remove_comma(row[1])])

# convert data into tensor
y = torch.tensor(y, dtype = torch.float32)
x1 = torch.tensor(x1, dtype = torch.float32)
x2 = torch.tensor(x2, dtype = torch.float32)
x3 = torch.tensor(x3, dtype = torch.float32)

# initialize the weights
w1 = torch.randn(1, requires_grad = True)
w2 = torch.randn(1, requires_grad = True)
w3 = torch.randn(1, requires_grad = True)
w4 = torch.randn(1, requires_grad = True)

# set learning rate and number of epochs
learning_rate = 1e-8
epochs = 100
losses = []

# train
for epoch in range(epochs):
    y_pred = torch.matmul(x1, w1) + torch.matmul(x2, w2) + torch.matmul(x3, w3) + w4
    loss = torch.mean((y_pred - y)**2)

    loss.backward()
    losses.append(loss.item())

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w3 -= learning_rate * w2.grad
        w4 -= learning_rate * w2.grad
        
        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
        w4.grad.zero_()

    if epoch in range(10) or epoch % 10 == 9:
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

print(f"Trained weights: w1 = {w1.item()}, w2 = {w2.item()}, w3 = {w3.item()}, w4 = {w4.item()}")