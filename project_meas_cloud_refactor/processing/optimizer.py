"""
Objective: Create optimization class that defines the fitting routine
"""

import torch
import numpy as np

from utils.optimizer_utils import condition_domain, dtime_loss_fn, pois_loss_fn, calc_rel_step

class Optimizer(torch.nn.Module):
    def __init__(self, degree, t):
        # Constants
        super().__init__()
        self.degree = degree  # Polynomial order
        self.C = torch.nn.Parameter(torch.zeros(degree + 1, dtype=float))
        self.B = torch.nn.Parameter(torch.ones(1, dtype=float))  # Background term
        self.t_cheby = condition_domain(t, t[0], t[-1], self.degree)

    def forward(self):
        # Evaluate polynomial
        poly = self.t_cheby @ self.C
        lamb = torch.exp(poly) + self.B

        return lamb

def optimize(
        t,
        Y_train,
        Y_val,
        Z_train,
        Z_val,
        Nshots_train,
        Nshots_val,
        degree,
        deadtime,
        learning_rate,
        rel_step_lim,
        max_epochs,
        term_persist
):
    print('Computing for degree: {}'.format(degree))
    dr_t = torch.diff(t)[0]  # [s] range resolution in time

    model = Optimizer(degree=degree, t=t)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 0
    rel_step = 1e3 * rel_step_lim

    rel_step_lst = []
    fit_loss_lst = []
    while (rel_step > rel_step_lim) and (epoch < max_epochs):
        optimizer.zero_grad()
        lamb = model()  # (M,)

        if deadtime:
            loss = dtime_loss_fn(lamb, Z_train, Y_train, Nshots_train, dr_t)
        else:
            loss = pois_loss_fn(lamb, Y_train, Nshots_train, dr_t)
        fit_loss_lst += [loss.item()]

        if epoch % 100 == 0:
            print(f"step {epoch}, loss = {loss.item():.4f}")

        rel_step_lst, rel_step = calc_rel_step(epoch, rel_step_lst, rel_step, fit_loss_lst, term_persist)

        loss.backward()
        optimizer.step()

        epoch += 1

    print('Exited process at epoch {}/{}'.format(epoch, max_epochs))
    with torch.no_grad():
        lamb_out = model()
        if deadtime:
            loss_val = dtime_loss_fn(lamb_out, Z_val, Y_val, Nshots_val, dr_t)
        else:
            loss_val = pois_loss_fn(lamb_out, Y_val, Nshots_val, dr_t)

    return (
        lamb_out.detach().cpu().numpy(),
        model.C.detach().cpu().numpy(),
        model.B.detach().cpu().numpy(),
        fit_loss_lst,
        loss_val.item()
    )
