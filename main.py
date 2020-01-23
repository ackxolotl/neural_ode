import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import random


class TrueLinearRHS(nn.Module):
    def __init__(self, A):
        super(TrueLinearRHS, self).__init__()
        self.A = A

    def forward(self, t, y):
        return torch.mv(self.A, y)


class AndronovHopfRHS(nn.Module):
    def __init__(self, alpha):
        super(AndronovHopfRHS, self).__init__()
        self.alpha = alpha

    def forward(self, t, y):
        y1 = self.alpha * y[0] - y[1] - y[0] * (y[0]**2 + y[1]**2)
        y2 = y[0] + self.alpha * y[1] - y[1] * (y[0]**2 + y[1]**2)
        return torch.tensor([y1, y2])


class NonlinearRHS(nn.Module):
    def __init__(self):
        super(NonlinearRHS, self).__init__()

    def forward(self, t, y):
        dy1 = y[0]**2 + y[1]**2
        dy2 = (y[0] - y[1])**3
        return torch.tensor([dy1, dy2])


class LinearRHS(nn.Module):
    def __init__(self, dim):
        super(LinearRHS, self).__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, t, y):
        return self.linear(y)


class MultiLayerRHS(nn.Module):
    def __init__(self, dim, hidden_dim=32):
        super(MultiLayerRHS, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, t, y):
        return self.layers(y)


def create_trajectory(rhs, y0, eval_times):
    return odeint(rhs, y0, eval_times)


def create_vector_field(rhs, y1_min, y1_max, y2_min, y2_max):
    Y1 = np.linspace(y1_min, y1_max)
    Y2 = np.linspace(y2_min, y2_max)
    y1s, y2s = len(Y1), len(Y2)
    Y1, Y2 = np.meshgrid(Y1, Y2)
    dY1, dY2 = np.zeros_like(Y1), np.zeros_like(Y2)
    for i in range(y1s):
        for j in range(y2s):
            y = torch.tensor([Y1[i, j], Y2[i, j]])
            dy = rhs(0, y)
            dY1[i, j] = dy[0]
            dY2[i, j] = dy[1]
    return Y1, Y2, dY1, dY2


def extract_vector_field(rhs_true, rhs_model, y0, T, num_evals, num_iterations=2000):
    # Create true trajectory
    eval_times = torch.linspace(0, T, num_evals)
    Y_true = create_trajectory(rhs_true, y0, eval_times)
    orbit = Y_true.detach().numpy()

    # Learn vector field by trying to approximate trajectory
    rhs_model = learn_dynamics_from_trajectory(rhs_model, [Y_true], eval_times, num_iterations)

    # Create predicted trajectory
    Y_pred = create_trajectory(rhs_model, y0, eval_times)
    orbit_pred = Y_pred.detach().numpy()

    y1_max = max(np.max(np.abs(orbit[:, 0])), np.max(np.abs(orbit_pred[:, 0])))
    y2_max = max(np.max(np.abs(orbit[:, 1])), np.max(np.abs(orbit_pred[:, 1])))
    y_max = max(y1_max, y2_max)

    # Create true vector field
    Y1, Y2, dY1, dY2 = create_vector_field(
        rhs_true,
        -y_max,
        y_max,
        -y_max,
        y_max
    )

    # Create predicted vector field
    Y3, Y4, dY3, dY4 = create_vector_field(
        rhs_model,
        -y_max,
        y_max,
        -y_max,
        y_max
    )
    # Print learned weights
    for p in rhs_model.parameters():
        print(p)

    # Plot results
    fig, (ax_true, ax_pred) = plt.subplots(nrows=1, ncols=2)
    ax_true.streamplot(Y1, Y2, dY1, dY2)
    ax_true.plot(orbit[:, 0], orbit[:, 1], 'r', label='True trajectory')
    ax_pred.streamplot(Y3, Y4, dY3, dY4)
    ax_pred.plot(orbit_pred[:, 0], orbit_pred[:, 1], 'r', label='Approximated trajectory')
    ax_true.set_title('True vector field')
    ax_pred.set_title('Vector field learned by NODE')
    for ax in [ax_true, ax_pred]:
        ax.set_xlabel(r'$y_1$')
        ax.set_ylabel(r'$y_2$')
        ax.set_aspect('equal')
        ax.legend()
    plt.tight_layout()
    plt.show()


def fit_synthetic_data(rhs_true, rhs_model, initial_points, T, num_evals, num_iterations=2000):
    # Create true trajectories
    eval_times = torch.linspace(0, T, num_evals)
    trajectories = [create_trajectory(rhs_true, y0, eval_times) for y0 in initial_points]

    # Learn vector field by trying to approximate trajectory
    rhs_model = learn_dynamics_from_trajectory(rhs_model, trajectories, eval_times, num_iterations)

    # Create predicted trajectory
    y1_max = 5
    y2_max = 5

    # Create true vector field
    Y1, Y2, dY1, dY2 = create_vector_field(
        rhs_true,
        -y1_max,
        y1_max,
        -y2_max,
        y2_max
    )

    # Create predicted vector field
    Y3, Y4, dY3, dY4 = create_vector_field(
        rhs_model,
        -y1_max,
        y1_max,
        -y2_max,
        y2_max
    )

    # Plot results
    fig, (ax_true, ax_pred) = plt.subplots(nrows=1, ncols=2)
    ax_true.streamplot(Y1, Y2, dY1, dY2)
    ax_true.set_xlim((-y1_max, y1_max))
    ax_true.set_ylim((-y2_max, y2_max))
    ax_pred.set_xlim((-y1_max, y1_max))
    ax_pred.set_ylim((-y2_max, y2_max))
    for Y in trajectories:
        ax_true.plot(Y[:, 0], Y[:, 1], 'r')
    trajectories = [create_trajectory(rhs_model, y0, eval_times) for y0 in initial_points]
    for Y in trajectories:
        ax_pred.plot(Y.detach().numpy()[:, 0], Y.detach().numpy()[:, 1], 'r')
    ax_pred.streamplot(Y3, Y4, dY3, dY4)
    ax_true.set_title('True vector field')
    ax_pred.set_title('Vector field learned by NODE')
    for ax in [ax_true, ax_pred]:
        ax.set_xlabel(r'$y_1$')
        ax.set_ylabel(r'$y_2$')
        ax.set_aspect('equal')
        ax.legend()
    plt.show()


def learn_dynamics_from_trajectory(rhs_model, trajectories, eval_times, num_iterations=2000):
    batch_size = 16
    batch_steps = 10
    optimizer = torch.optim.Adam(rhs_model.parameters())

    for iteration in range(num_iterations):
        Y_true = random.choice(trajectories)
        indices = torch.tensor(random.sample(range(len(Y_true) - batch_steps), batch_size))
        Y0_batch = Y_true[indices]
        Y_true_batch = torch.stack([Y_true[indices + i] for i in range(batch_steps)], dim=0)
        batch_times = eval_times[:batch_steps]
        optimizer.zero_grad()
        Y_pred = odeint(rhs_model, Y0_batch, batch_times)
        loss = nn.functional.mse_loss(Y_pred, Y_true_batch)
        if iteration % 500 == 0:
            print(f'Iteration {iteration}: loss = {loss}')
        loss.backward()
        optimizer.step()

    return rhs_model


def main():
    # # Linear RHS
    # # Extraction of linear vector field (spiral) --> works
    # A = torch.tensor([[0.1, 0.1], [-0.25, 0.0]])
    # y0 = torch.tensor([1.0, 0.0])
    # T = 25
    # num_evals = 1000
    # extract_vector_field(TrueLinearRHS(A), LinearRHS(2), y0, T, num_evals)
    #
    # # Extraction of other linear vector field --> also works
    # A = torch.tensor([[0.1, 0], [0.0, -0.1]])
    # y0 = torch.tensor([0.1, 0.1])
    # T = 25
    # num_evals = 1000
    # extract_vector_field(TrueLinearRHS(A), LinearRHS(2), y0, T, num_evals)

    # # Extraction of nonlinear vector field (Andronov-Hopf) --> doesn't work with linear rhs
    # alpha = 1
    # y0 = torch.tensor([0.1, 0.1])
    # T = 25
    # num_evals = 1000
    # extract_vector_field(AndronovHopfRHS(alpha), LinearRHS(2), y0, T, num_evals)


    y0 = torch.tensor([-2.0, 2.0])
    T = 0.5
    num_evals = 1000
    extract_vector_field(NonlinearRHS(), MultiLayerRHS(2), y0, T, num_evals)

    # Nonlinear RHS (2-layer MLP)
    # A = torch.tensor([[0.1, 0.1], [-0.25, 0.0]])
    # y0 = torch.tensor([1.0, 0.0])
    # T = 25
    # num_evals = 1000
    # extract_vector_field(TrueLinearRHS(), MultiLayerRHS(2), y0, T, num_evals, 2000)
    #
    # A = torch.tensor([[0.1, 0], [0.0, -0.1]])
    # y0 = torch.tensor([0.1, 0.1])
    # T = 25
    # num_evals = 1000
    # extract_vector_field(TrueLinearRHS(A), MultiLayerRHS(2), y0, T, num_evals, 10000)

    # alpha = 1
    # y0 = torch.tensor([0.1, 0.1])
    # T = 25
    # num_evals = 1000
    # extract_vector_field(AndronovHopfRHS(alpha), MultiLayerRHS(2), y0, T, num_evals, 10000)


def train_nonlinear_rhs():
    num_initial_points = 64
    # initial_points = [
    #     5 * (torch.rand(2) - 0.5 * torch.ones(2)) for _ in range(num_initial_points)
    # ]
    initial_points = [
        # torch.tensor([0.1, 0.1]),
        # torch.tensor([0.1, -0.1]),
        # torch.tensor([-0.1, 0.1]),
        # torch.tensor([-0.1, -0.1]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, -2.0]),
        torch.tensor([-1.0, 2.0]),
        torch.tensor([-1.0, -2.0]),
        torch.tensor([0.0, 2.0]),
        torch.tensor([0.0, -2.0])
        # torch.tensor([2.0, 2.0]),
        # torch.tensor([2.0, -2.0]),
        # torch.tensor([-2.0, 2.0]),
        # torch.tensor([1.5, -1.5]),
        # torch.tensor([1.0, 1.0]),
        # torch.tensor([-4.0, -4.0])
    ]
    # A = torch.tensor([[0.1, 0.1], [-0.25, 0.0]])
    # T = 25
    # num_evals = 1000
    # fit_synthetic_data(TrueLinearRHS(A), MultiLayerRHS(2), initial_points, T, num_evals, 10000)

    A = torch.tensor([[0.1, 0], [0.0, -0.1]])
    T = 25
    num_evals = 1000
    fit_synthetic_data(TrueLinearRHS(A), MultiLayerRHS(2), initial_points, T, num_evals, 2000)

    # alpha = 1
    # T = 25
    # num_evals = 1000
    # fit_synthetic_data(AndronovHopfRHS(alpha), MultiLayerRHS(2), initial_points, T, num_evals, 10000)

    # T = 25
    # num_evals = 1000
    # fit_synthetic_data(NonlinearRHS(), MultiLayerRHS(2), initial_points, T, num_evals, 2000)

    # T = 0.5
    # num_evals = 1000
    # fit_synthetic_data(NonlinearRHS(), MultiLayerRHS(2), initial_points, T, num_evals, 20000)


if __name__ == '__main__':
    # main()
    train_nonlinear_rhs()
