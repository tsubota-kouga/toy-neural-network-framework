import numpy as np
import pandas as pd
from tqdm import trange
from matplotlib import pyplot as plt
from matplotlib import cm

from nn.activation import ReLU, Sigmoid, Step
from nn.basis import Sequential, Variable
from nn.linear import Linear
from nn.loss import BCELoss, MSELoss
from nn.optimizer import SGD


NUM_EPOCHS = 200
LR = 0.1


if __name__ == "__main__":
    model = Sequential(
        Linear(in_features=2, out_features=8),
        Sigmoid(),
        Linear(in_features=8, out_features=1),
        Sigmoid())
    criterion = MSELoss()

    df = pd.read_table("./data/data3.csv")
    assert isinstance(df, pd.DataFrame)
    data: np.ndarray = df[["X1", "X2"]].to_numpy()
    label: np.ndarray = df[["T"]].to_numpy()

    sgd = SGD(model.parameters(), lr=LR)

    progress_bar = trange(NUM_EPOCHS)
    loss_history: list[float] = []
    for _ in progress_bar:
        losses = 0.
        for d, l in zip(data[:, None], label[:, None]):
            y = model.forward(Variable(d))
            loss = criterion.forward(y, Variable(l))
            loss.backward()
            losses += loss.mean(axis=0).data.item()
            sgd.step()
            loss.zero_grad()
        losses = losses / len(data)
        progress_bar.set_description(f"loss:{losses}")
        loss_history.append(losses)
    plt.plot(list(range(len(loss_history))), loss_history)
    plt.savefig("output2-1.jpg")
    plt.show()

    predict = (model.forward(Variable(data)).data > 0.5).astype(np.int64)
    predict = predict.reshape(-1)
    label = label.reshape(-1)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        data[np.logical_and(predict == label, label == 0), 0],
        data[np.logical_and(predict == label, label == 0), 1],
        np.full_like(data[np.logical_and(predict == label, label == 0), 0], 0.5),
        marker="o", c="green")
    ax.scatter(
        data[np.logical_and(predict == label, label == 1), 0],
        data[np.logical_and(predict == label, label == 1), 1],
        np.full_like(data[np.logical_and(predict == label, label == 1), 0], 0.5),
        marker="o", c="blue")
    ax.scatter(
        data[predict != label, 0],
        data[predict != label, 1],
        np.full_like(data[predict != label, 1], 0.5),
        marker="x", c="red")

    X: np.ndarray = np.arange(-1., 2., 0.1)
    Y: np.ndarray = np.arange(-1., 2., 0.1)
    size = X.size
    X, Y = np.meshgrid(X, Y)
    XY = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    Z = model.forward(Variable(XY))

    theCM = cm.get_cmap()
    theCM._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
    theCM._lut[:-3,-1] = alphas

    ax.plot_surface(
        X,
        Y,
        Z.data.reshape(size, size),
        cmap=theCM)

    plt.savefig("output2-2.jpg")
    plt.show()
