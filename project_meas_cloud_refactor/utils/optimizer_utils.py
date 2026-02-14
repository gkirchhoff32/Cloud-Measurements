"""
Define functions that are used in the optimizer class
"""

import torch
import numpy as np

def dtime_loss_fn(lamb, Z, Y, Nshots, dt):
    # Deadtime loss function (discrete)
    return (Nshots * lamb * Z * dt - Y * torch.log(lamb)).sum()

def pois_loss_fn(lamb, Y, Nshots, dt):
    # Poisson loss function (discrete)
    return (Nshots * lamb * dt - Y * torch.log(lamb)).sum()

def condition_domain(t, t_min, t_max, degree):
    t_scaled = 2 * (t - t_min) / (t_max - t_min) - 1
    t_cheby = cheby_poly(t_scaled, degree)

    return t_cheby

def cheby_poly(x, degree):
    def cheby(x, m):
        T0 = x ** 0
        T1 = x ** 1
        if m == 0:
            return T0
        elif m == 1:
            return T1
        else:
            return (2 * x * cheby(x, m-1) - cheby(x, m-2))

    N = len(x)
    model_out = torch.zeros((N, degree+1), dtype=float)
    for i in range(degree+1):
        model_out[:, i] = cheby(x, i)

    return model_out

def calc_rel_step(epoch, rel_step_lst, rel_step, fit_loss_lst, term_persist):
    if epoch == 0:
        rel_step_lst += [rel_step]
    else:
        rel_step_lst += [(fit_loss_lst[-2] - fit_loss_lst[-1]) / np.abs(fit_loss_lst[-2])]
        rel_step = np.abs(np.array(rel_step_lst)[-term_persist:].mean())

    return rel_step_lst, rel_step