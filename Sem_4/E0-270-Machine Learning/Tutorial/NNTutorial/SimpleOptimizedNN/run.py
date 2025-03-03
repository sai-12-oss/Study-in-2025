import numpy as np

from model import Network
from utils import ArgumentStorage, plot_decision_boundary


def main(args: ArgumentStorage) -> None:
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    Y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    model = Network(args.num_inputs, args.num_hidden, args.num_outputs)

    for i in range(args.num_iters + 1):
        idxs = np.random.choice(len(X), args.batch_size, replace=False)
        x = X[idxs]
        y = Y[idxs]

        y_pred = model.forward(x)

        loss = np.square(y_pred - y).mean()
        grad_L = 2 * (y_pred - y) / len(y)

        model.zero_grad()
        model.backward(grad_L)
        model.update(args.lr)

        if i % 100 == 0:
            print(f'Epoch: {i}, Loss: {loss}')

    print('Final predictions:')
    for x, y in zip(X, Y):
        y_pred = model.forward(x)
        print(f'x: {x}, y: {y}, y_pred: {y_pred}')

    plot_decision_boundary(model, X, Y)


if __name__ == '__main__':
    args = ArgumentStorage({
        'num_inputs': 2,
        'num_hidden': 3,
        'num_outputs': 1,
        'lr': 1e-2,
        'num_iters': 5000,
        'batch_size': 2,
    })
    main(args)
