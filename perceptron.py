import numpy as np
import pandas as pd
from tqdm import trange
from matplotlib import pyplot as plt

from nn.activation import ReLU, Sigmoid, Step
from nn.basis import Sequential, Variable
from nn.linear import Linear
from nn.loss import BCELoss, MSELoss
from nn.optimizer import SGD


NUM_EPOCHS = 50
LR = 0.005

NUM_ITER = 5
COLORS = ["red", "green", "blue", "pink", "purple"]


if __name__ == "__main__":

    for i in trange(NUM_ITER):
        model = Sequential(
            Linear(in_features=2, out_features=1),
            Step())
        criterion = MSELoss()

        model.modules.modules[0].weight.variable.data[0, 0] = 0.1 * i ** 2 - 0.8
        model.modules.modules[0].weight.variable.data[1, 0] = -0.01 * i ** 4 - 1.

        df = pd.read_table("./data/data1.csv")
        assert isinstance(df, pd.DataFrame)
        data = df[["x1", "x2"]].to_numpy()
        label = df[["Class"]].to_numpy()

        sgd = SGD(model.parameters(), lr=LR)

        progress_bar = trange(NUM_EPOCHS)
        w1_over_w2, b_over_w2 = [], []
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
            w1, w2 = model.modules.modules[0].weight.variable.data.reshape(-1).tolist()
            b = model.modules.modules[0].bias.variable.data.item()
            w1_over_w2.append(w1 / w2)
            b_over_w2.append(b / w2)

        if i == 0:
            plt.plot(list(range(len(loss_history))), loss_history)
            plt.savefig("output1-1.jpg")
            plt.show()

            plt.plot(list(range(len(w1_over_w2))), w1_over_w2, color="red", label="w1/b")
            plt.plot(list(range(len(b_over_w2))), b_over_w2, color="blue", label="w2/b")
            plt.legend()
            plt.savefig("output1-2.jpg")
            plt.show()

            w1, w2 = model.modules.modules[0].weight.variable.data.reshape(-1).tolist()
            b = model.modules.modules[0].bias.variable.data.item()

        plt.quiver(
            np.array(w1_over_w2[:-1]),
            np.array(b_over_w2[:-1]),
            np.array(w1_over_w2[1:]) - np.array(w1_over_w2[:-1]),
            np.array(b_over_w2[1:]) - np.array(b_over_w2[:-1]),
            color=COLORS[i],
            scale_units='xy', angles='xy', scale=1, label=f"(w1/w2, b/w2) (i={i})")
        if i == 4:
            plt.legend()
            plt.savefig("output1-3.jpg")
            plt.show()
