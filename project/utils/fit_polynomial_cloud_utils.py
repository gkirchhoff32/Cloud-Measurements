# fit_polynomial_utils.py
#
# Methods for deadtime evaluation scripts.
#
# Grant Kirchhoff
# Last Updated: 02/02/2023

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# build the fit model as a NN module
class Fit_Pulse(torch.nn.Module):
    def __init__(self, M, t_min, t_max):
        """
        Instantiate and initialize the fit parameters.
        :param M: (int) Polynomial order
        :param t_min: (float) Lower bound for fit window [s]
        :param t_max: (float) Upper bound for fit window [s]
        """
        super().__init__()
        self.M = M  # Polynomial order
        self.C = torch.nn.Parameter(-1 * torch.ones(M+1, 1, dtype=float))  # Coefficients to be optimized
        self.B = torch.nn.Parameter(torch.ones(1, dtype=float))
        # self.B = 0
        self.t_max = t_max  # Fit upper bound
        self.t_min = t_min  # Fit lower bound

    # Helpers for numerical integration (Riemann and trapezoidal method)
    @staticmethod
    def trapezoid(vals, dx):
        trap_intgrl = 2*torch.sum(vals) - vals[0] - vals[-1]
        trap_intgrl *= dx / 2
        return trap_intgrl

    @staticmethod
    def riemann(vals, dx):
        riem_intgrl = torch.sum(vals) * dx
        return riem_intgrl

    def tstamp_condition(self, t, t_min, t_max):
        """
        Transform time tag array into chebyshev polynomial matrix form.
        :param t: (torch.tensor) Time tag array [s]
        :param t_min: (float) Lower bound for fit window [s]
        :param t_max: (float) Upper bound for fit window [s]
        :return:
        t_poly_cheb: (torch.tensor) Chebyshev polynomial matrix of time tags
        """
        t_norm = (t - t_min) / (t_max - t_min)  # Normalize timestamps along [0,1]
        t_poly_cheb = cheby_poly(t_norm, self.M)  # Generate chebyshev timestamp basis
        return t_poly_cheb

    def forward(self, intgrl_N, active_ratio_hst, t, t_intgrl, discrete_loss, cheby=True):
        """
        Forward model the profile for input time t of polynomial order M (e.g., x^2 --> M=2).
        Also return the integral.
        Parameters:
        intgrl_N  (int): number of steps in numerical integration \\ []
        active_ratio_hst (torch array): Deadtime-adjusted array ("deadtime_adjust_vals output") \\ [Nx1]
        t (torch array): time stamps (unnormalized if cheby=False, cheby_poly output if cheby=True) \\ [Nx1]
        t_N (float): maximum time stamp value \\ []
        t_intgrl (torch array): time vector [0,1] as chebyshev polynomial (i.e., cheby_poly output) \\ [intgrl_Nx1]
        discrete_loss (bool): if using discrete or continuous time-tag form of loss function
        cheby (bool): Set true if t is normalized (i.e., output from self.tstamp_condition)
        Returns:
        model_out    (torch array): forward model                    \\ [Nx1]
        integral_out (torch array): finite numerical integral output \\ float
        """

        # orthonormalize by leveraging chebyshev polynomials, then calculate forward model
        if not cheby:
            # t_fit_norm = fit_model.tstamp_condition(t_phot_fit_tnsr, t_min, t_max)
            t_poly_cheb = self.tstamp_condition(t, self.t_min, self.t_max)
        else:
            t_poly_cheb = t

        if not discrete_loss:
            poly = t_poly_cheb @ self.C
            model_out = torch.exp(poly) + self.B  # Forward model from time tags

            # calculate the integral
            poly = t_intgrl @ self.C
            fine_res_model = torch.exp(poly) + self.B

            t_fine, dt = np.linspace(self.t_min, self.t_max, intgrl_N, endpoint=False, retstep=True)
            active_ratio_hst.resize_(fine_res_model.size())
            fine_res_model = fine_res_model * active_ratio_hst  # Generate deadtime noise model
            integral_out = self.riemann(fine_res_model, dt)

            return model_out, integral_out

        else:
            poly = t_intgrl @ self.C
            fine_res_model = torch.exp(poly) + self.B
            fine_res_model = torch.reshape(fine_res_model, active_ratio_hst.size())
            t_fine, dt = np.linspace(self.t_min, self.t_max, intgrl_N+1, endpoint=True, retstep=True)

            return fine_res_model, dt


def pois_loss_discrete(pred_fit, eta, active_ratio_hst, dt, Y, Nshots):
    """
    Non-homogenous Poisson point process loss function - discrete form
    """
    loss = torch.sum(Nshots*eta*pred_fit*active_ratio_hst*dt - Y*torch.log(eta*pred_fit))

    return loss

def pois_loss_time_tag(prof, integral, n_shots, eta):
    """
    Non-homogenous Poisson point process loss function - time-tag form
    """
    loss = n_shots*eta*integral - torch.sum(torch.log(eta*prof))

    return loss

# Chebyshev polynomial matrix generator
def cheby_poly(x, M):
    """
    Parameters:
    x (torch array): Values to be evaluated on in chebyshev polynomial      \\ [Nx1]
    M (int)        : *Highest* order term of polynomial (e.g., x^2 --> M=2) \\ []
    Returns:
    chebyshev polynomial matrix (torch array): Evaluated polynomial \\ [NxM]
    """

    def cheby(x, m):
        """
        Helper to calculate value of specific chebyshev order
        """
        T0 = x ** 0
        T1 = x ** 1
        if m == 0:
            return T0
        elif m == 1:
            return T1
        else:
            return 2 * x * cheby(x, m - 1) - cheby(x, m - 2)

    N = len(x)
    model_out = torch.zeros((N, M+1), dtype=float)
    for i in range(M + 1):
        model_out[:, i] = cheby(x, i)

    return model_out

# Deadtime Noise Model
#
# Adjust bin ratios depending on reduced bin availability due to deadtime. This is done because deadtime reduces
# available bins following a detection event. To accommodate for this, in the loss function the impact each time bin has
# on the numerical integration is proportionally reduced to how long it was active (i.e., unaffected by deadtime).
# Please refer to Willem Marais's and/or Matt Hayman's notes on this subject.

def deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst, n_shots):
    """
    Deadtime adjustment for arrival rate estimate in optimizer.
    Parameters:
    t_min: Window lower bound \\ float [s]
    t_max: Window upper bound \\ float [s]
    intgrl_N (int): Number of bins in integral \\ int
    deadtime: Deadtime interval [sec] \\ float
    t_det_lst (list): Nested list of arrays, where each array contains the detections per laser shot
    Returns:
    active_ratio_hst (torch array): Histogram of deadtime-adjustment ratios for each time bin.
    """

    # Initialize
    bin_edges, dt = np.linspace(t_min, t_max, intgrl_N+1, endpoint=True, retstep=True)
    active_ratio_hst = n_shots * np.ones(len(bin_edges)-1)
    deadtime_n_bins = np.ceil(deadtime / dt).astype(int)  # Number of bins that deadtime occupies

    # Iterate through each shot. For each detection event, reduce the number of active bins according to deadtime length.
    for shot in range(len(t_det_lst)):
        detections = t_det_lst[shot]

        for det in detections:
            det_time = det.item()  # Time tag of detection that occurred during laser shot

            # Only include detections that fall within fitting window
            if det_time >= (t_min-deadtime) and det_time <= t_max:
                det_bin_idx = np.argmin(abs(det_time - bin_edges))  # Bin that detection falls into
                if det_time < t_min:
                    deadtime_n_bins_adjusted = deadtime_n_bins - np.ceil(t_min-det_time / dt).astype(int)
                    final_dead_bin = det_bin_idx + deadtime_n_bins_adjusted
                else:
                    final_dead_bin = det_bin_idx + deadtime_n_bins  # Final bin index that deadtime occupies

                # Currently a crutch that assumes "dead" time >> potwindow. Will need to include "wrap around" to be more accurate
                # If final dead bin surpasses fit window, set it to the window upper bin
                if final_dead_bin > len(active_ratio_hst):
                    final_dead_bin = len(active_ratio_hst)
                # If initial dead bin (detection bin) precedes fit window, set it to the window lower bin
                if det_time < t_min:
                    det_bin_idx = 0
                active_ratio_hst[det_bin_idx:final_dead_bin] -= 1  # Remove "dead" region in active ratio

    # active_ratio_hst /= len(t_det_lst)  # Normalize for ratio
    active_ratio_hst /= n_shots  # Normalize for ratio

    # fig = plt.figure(figsize=(8, 4), dpi=400)
    # ax = fig.add_subplot(111)
    # ax.plot(bin_edges[:-1]*3e8/2, active_ratio_hst)
    # ax.set_xlabel('Range [m]')
    # ax.set_ylabel('AF Histogram')
    # plt.show()
    # quit()

    return torch.tensor(active_ratio_hst), bin_edges

def generate_fit_val(data, t_det_lst, n_shots):
    """
    Generates fit, validation, and evaluation data sets for the fitting routine. For reference - (1) Fit set: Dataset
    used to generate the fit; (2) Validation set: Independent dataset used to calculate validation loss; and (3)
    Evaluation set: High-fidelity set (e.g., unaffected by deadtime, high-OD setting) that is used to calculate
    evaluation loss.

    :param data: (Nx1) Data used for calculating fit and validation loss
    :param data_ref: (Mx1) Reference data used for calculating evaluation loss
    :param n_shots: (int) Number of laser shots for "data"
    :param n_shots_ref: (int) Number of laser shots for "data_ref"
    :return: t_phot_fit_tnsr: (N/2 x 1) Fit set (torch tensor)
    :return: t_phot_val_tnsr: (N/2 x 1) Validation set (torch tensor)
    :return: t_phot_eval_tnsr: (Mx1) Evaluation set (torch tensor)
    :return: n_shots_fit: Number of laser shots for fit set
    :return: n_shots_val: Number of laser shots for validation set
    :return: n_shots_eval: Number of laser shots for evaluation set
    """

    # The target is assumed to be stationary, so I can split the data into halves. Proof: Poisson thinning.
    split_value = int(len(data) // 2)
    t_phot_fit = data[:split_value]
    t_phot_val = data[split_value:]
    # t_phot_eval = data_ref[:]
    split_value_det = int(len(t_det_lst) // 2)
    t_det_lst_fit = t_det_lst[:split_value_det]
    t_det_lst_val = t_det_lst[split_value_det:]

    # Adjust number of laser shots corresponding to fit and val sets
    ratio_fit_split = len(t_phot_fit) / len(data)
    ratio_val_split = len(t_phot_val) / len(data)
    n_shots_fit = round(n_shots * ratio_fit_split)
    n_shots_val = round(n_shots * ratio_val_split)
    # n_shots_eval = n_shots_ref

    if isinstance(t_phot_fit, np.ndarray):
        t_phot_fit_tnsr = torch.tensor(t_phot_fit)
        t_phot_val_tnsr = torch.tensor(t_phot_val)
    else:
        t_phot_fit_tnsr = torch.tensor(t_phot_fit.to_numpy())
        t_phot_val_tnsr = torch.tensor(t_phot_val.to_numpy())
    # t_phot_eval_tnsr = torch.tensor(t_phot_eval.to_numpy())

    return t_phot_fit_tnsr, t_phot_val_tnsr, t_det_lst_fit, t_det_lst_val, n_shots_fit, n_shots_val


# Generate fit routine
def optimize_fit(M_max, M_lst, t_fine, t_phot_fit_tnsr, t_phot_val_tnsr, active_ratio_hst_fit,
                active_ratio_hst_val, n_shots_fit, n_shots_val, T_BS, use_com_det, discrete_loss=False, learning_rate=1e-1,
                rel_step_lim=1e-8, intgrl_N=10000, max_epochs=4000, term_persist=20):

    t_min, t_max = t_fine[0], t_fine[-1]
    bins = np.append(t_fine, t_fine[-1] + np.diff(t_fine)[0])

    val_loss_arr = np.full(M_max+1, np.nan)
    coeffs = np.zeros((M_max+1, M_max+1))
    fit_rate_fine = np.zeros((M_max+1, len(t_fine)))
    C_scale_arr = np.zeros(M_max+1)
    print('Time elapsed:\n')

    if use_com_det == 0 or use_com_det == 2:
        T_BS_LG = T_BS[0]
        eta_LG = T_BS_LG
        t_phot_fit_tnsr_LG, t_phot_val_tnsr_LG = t_phot_fit_tnsr[0], t_phot_val_tnsr[0]
        active_ratio_hst_fit_LG, active_ratio_hst_val_LG = active_ratio_hst_fit[0], active_ratio_hst_val[0]
        n_shots_fit_LG, n_shots_val_LG = n_shots_fit[0], n_shots_val[0]
        Y_fit_LG = torch.from_numpy(np.histogram(t_phot_fit_tnsr_LG.detach().numpy(), bins=bins)[0])
        Y_val_LG = torch.from_numpy(np.histogram(t_phot_val_tnsr_LG.detach().numpy(), bins=bins)[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Iterate through increasing polynomial complexity.
    # Compare fit w/ validation set and use minimum loss find optimal polynomial order.
    for i in range(len(M_lst)):
        # initialize for fit loop
        M = M_lst[i]  # Polynomial order  (e.g., x^2 --> M=2)
        fit_model = Fit_Pulse(M, t_min, t_max)
        optimizer = torch.optim.Adam(fit_model.parameters(), lr=learning_rate)
        epoch = 0
        rel_step = 1e3 * rel_step_lim
        fit_loss_lst = []
        rel_step_lst = []

        init_C = np.zeros(M + 1)
        for j in range(M + 1):
            init_C[j] = fit_model.C[j].item()

        # set the loss function to use a Poisson point process likelihood function
        if discrete_loss:
            loss_fn = pois_loss_discrete
        else:
            loss_fn = pois_loss_time_tag

        # perform fit
        start = time.time()
        t_fit_norm_LG = fit_model.tstamp_condition(t_phot_fit_tnsr_LG, t_min, t_max)
        t_val_norm_LG = fit_model.tstamp_condition(t_phot_val_tnsr_LG, t_min, t_max)

        t_intgrl = cheby_poly(torch.linspace(0, 1, intgrl_N), M)
        while rel_step > rel_step_lim and epoch < max_epochs:
            fit_model.train()
            if discrete_loss:
                if use_com_det == 0 or use_com_det == 1:
                    fine_res_model_fit_LG, dt = fit_model(intgrl_N, active_ratio_hst_fit_LG, t_fit_norm_LG, t_intgrl, discrete_loss, cheby=True)
                    loss_fit_LG = loss_fn(fine_res_model_fit_LG, eta_LG, active_ratio_hst_fit_LG, dt, Y_fit_LG, n_shots_fit_LG)  # add regularization here
            else:
                if use_com_det == 0 or use_com_det == 2:
                    pred_fit_LG, integral_LG = fit_model(intgrl_N, active_ratio_hst_fit_LG, t_fit_norm_LG, t_intgrl, discrete_loss, cheby=True)
                    loss_fit_LG = loss_fn(pred_fit_LG, integral_LG, n_shots_fit_LG, eta_LG)

            if use_com_det == 0:
                loss_fit = loss_fit_LG
            # elif use_com_det == 1:
            #     loss_fit = loss_fit_HG
            # elif use_com_det == 2:
            #     loss_fit = loss_fit_LG/torch.sum(Y_fit_LG) + loss_fit_HG/torch.sum(Y_fit_HG)

            fit_loss_lst += [loss_fit.item()]

            # calculate relative step as an average over the last term_persist iterations
            if epoch == 0:
                rel_step_lst += [1e3 * rel_step_lim]
                rel_step = 1e3 * rel_step_lim
            else:
                rel_step_lst += [(fit_loss_lst[-2] - fit_loss_lst[-1]) / np.abs(fit_loss_lst[-2])]
                rel_step = np.abs(np.array(rel_step_lst)[-term_persist:].mean())

            # update estimated parameters
            loss_fit.backward()
            optimizer.step()

            # zero out the gradient for the next step
            optimizer.zero_grad()

            epoch += 1

        t_fine_tensor = torch.tensor(t_fine)
        pred_mod_seg, __ = fit_model(intgrl_N, active_ratio_hst_fit_LG, t_fine_tensor, t_intgrl, discrete_loss, cheby=False)
        fit_rate_fine[M, :] = pred_mod_seg.detach().numpy().T
        coeffs[M, 0:M + 1] = fit_model.C.detach().numpy().T

        # Calculate validation loss
        # Using fit generated from fit set, calculate loss when applied to validation set
        if discrete_loss:
            fine_res_model_val_LG, dt = fit_model(intgrl_N, active_ratio_hst_val_LG, t_val_norm_LG, t_intgrl, discrete_loss, cheby=True)
            loss_val_LG = loss_fn(fine_res_model_val_LG, eta_LG, active_ratio_hst_val_LG, dt, Y_val_LG, n_shots_val_LG)  # add regularization here
        else:
            pred_val_LG, integral_val_LG = fit_model(intgrl_N, active_ratio_hst_fit_LG, t_fit_norm_LG, t_intgrl, discrete_loss, cheby=True)
            loss_val_LG = loss_fn(pred_val_LG, integral_val_LG, n_shots_val_LG, eta_LG)

        loss_fit = loss_val_LG
        # elif use_com_det == 1:
        #     loss_fit = loss_val_HG
        # elif use_com_det == 2:
        #     loss_fit = loss_val_LG/torch.sum(Y_val_LG) + loss_val_HG/torch.sum(Y_val_HG)
        val_loss_arr[M] = loss_fit

        end = time.time()
        print('Order={}: {:.2f} sec'.format(M, end - start))

        ax.plot(fit_loss_lst, label='Order {}'.format(M))

    return ax, val_loss_arr, fit_rate_fine, coeffs







### Graveyard ###

# def loss_lse(f1, f2):
#     # LSE for 'C_optimize'
#     return 0.5*(f1 - f2)**2

# def C_optimize(loss1, loss_fn, ratio_step=0.99999, max_epochs=1000):
#     """
#     Calculate optimal scaling constant for arrival rate to calculate costs between two datasets with mismatching laser shots. For example, sets w/ different OD values are not comparable in the MLE loss function. There is a scaling that needs to happen to compare the two.
#     Parameters:
#     t_min: Window lower bound \\ float
#     t_max: Window upper bound \\ float
#     intgrl_N (int): Number of bins in integral \\ int
#     deadtime: Deadtime interval [sec] \\ float
#     t_det_lst (list): Nested list of arrays, where each array contains the detections per laser shot
#     Returns:
#     active_ratio_hst (torch array): Histogram of deadtime-adjustment ratios for each time bin.
#     """
#     epoch = 0
#     alpha = 0.00000000001
#     C = 1
#     while ratio_step<=0.99999 and epoch<max_epochs:
#         n_det_no_dtime = len(pred_no_dtime)
#         loss2 = loss_fn(C*pred_no_dtime, C*integral_no_dtime*n_shots_no_dtime)
#         cost = loss_lse(loss1, loss2)
#         step = -2*(loss1-loss2)*(n_shots_no_dtime*integral_no_dtime - n_det_no_dtime/C)
#         C = C - alpha*step

#         cost_lst.append(cost.item())
#         C_lst.append(C.item())

#         if epoch!=0:
#             ratio_step = C_lst[-1]/C_lst[-2]

#         epoch += 1

#     return C


