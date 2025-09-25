from sklearn.datasets import load_digits
import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 2, 1),
            nn.Tanh(),
            nn.Conv2d(16, 8, 3, 1, 1)
        )

        self.output = nn.Linear(32, 10)

    def forward(self, x):
        out = self.conv(x)
        out = self.output(out.flatten(1))
        return out

RATIO = 0.8
BATCH_SIZE = 128
EPOCH = 10

if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    X = X / 16.
    sample_num = len(y)
    X = [x.reshape(1, 8, 8).tolist() for x in X]

    indice = np.arange(sample_num)
    np.random.shuffle(indice)

    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    offline = int(sample_num * RATIO)

    train = Data.TensorDataset(X[indice[:offline]], y[indice[:offline]])
    test  = Data.TensorDataset(X[indice[offline:]], y[indice[offline:]])

    train_loader = Data.DataLoader(train, BATCH_SIZE, True)
    test_loader  = Data.DataLoader(test,  BATCH_SIZE, False)
    
    model = Digit()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    test_losses = []
    test_accs = []

    for epoch in range(EPOCH):
        model.train()
        for bx, by in train_loader:
            out = model(bx)
            loss = criterion(out, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        test_loss = []
        test_acc = []
        for bx, by in test_loader:
            with torch.no_grad():
                out = model(bx)
                pre_lab = out.argmax(1)
                loss = criterion(out, by)

            test_loss.append(loss.item())
            test_acc.append(accuracy_score(pre_lab, by))

        test_losses.append(np.mean(test_loss))
        test_accs.append(np.mean(test_acc))
    
    plt.figure(dpi=120)
    plt.plot(test_losses, 'o-', label="loss")
    plt.plot(test_accs, 'o-', label="accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    if not os.path.exists("model"):
        os.makedirs("model")
    torch.save(model.state_dict(), "model/digit.pth")

    # Convert to TorchScript via tracing
    model = Digit()
    model.load_state_dict(torch.load("model/digit.pth", map_location="cpu"))

    sample = torch.randn(1, 1, 8, 8)

    trace_model = torch.jit.trace(model, sample)
    trace_model.save("model/digit.jit")