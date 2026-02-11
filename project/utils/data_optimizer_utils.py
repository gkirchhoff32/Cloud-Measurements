import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
from pathlib import Path
import sys
import yaml

class DataOptimizer(torch.nn.Module):
    def __init__(self, degree, t):
        # Constants
        super().__init__()
        self.degree = degree  # Polynomial order
        # self.C = torch.nn.Parameter(-1 * torch.ones(degree+1, 1, dtype=float))  # Coefficients to optimize
        # self.C = torch.nn.Parameter(-1 * torch.zeros(degree+1, 1, dtype=float))  # Coefficients to optimize
        self.C = torch.nn.Parameter(torch.zeros(degree + 1, dtype=float))
        self.B = torch.nn.Parameter(torch.ones(1, dtype=float))  # Background term
        self.t_cheby = condition_domain(t, t[0], t[-1], self.degree)

    def forward(self):
        # Evaluate polynomial
        poly = self.t_cheby @ self.C
        lamb = torch.exp(poly) + self.B
        # lamb = torch.exp(poly)

        return lamb


def dtime_loss_fn(lamb, Z, Y, Nshots, dt):
    loss = (Nshots * lamb * Z * dt - Y * torch.log(lamb)).sum()

    return loss

def pois_loss_fn(lamb, Y, Nshots, dt):
    loss = (Nshots * lamb * dt - Y * torch.log(lamb)).sum()

    return loss

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

def condition_domain(t, t_min, t_max, degree):
    t_scaled = 2 * (t - t_min) / (t_max - t_min) - 1
    t_cheby = cheby_poly(t_scaled, degree)

    return t_cheby

def data_setup(loader, deadtime_correct, histogram_results):
    flux_raw = histogram_results['flux_raw']  # [Hz]
    cnts_raw = histogram_results['cnts_raw']
    af_results = deadtime_correct.calc_af_hist_convolution(histogram_results, loader)
    af_hist = af_results['af_hist']

    deadtime_trim_idx = deadtime_correct.deadtime_trim_idx
    flux_raw = flux_raw[deadtime_trim_idx:, :]  # [Hz] Removing initial loaded bins for AF hist calculation
    cnts_raw = cnts_raw[deadtime_trim_idx:, :]  # Removing initial loaded bins for AF hist calculation
    t_binedges = histogram_results['t_binedges']
    r_binedges = histogram_results['r_binedges'][deadtime_trim_idx:]

    flux_bin_est = flux_raw / af_hist

    plot_flux_est(flux_raw, flux_bin_est, t_binedges, r_binedges)

    return {'flux_raw': flux_raw,
            'cnts_raw': cnts_raw,
            'af_hist': af_hist,
            't_binedges': t_binedges,
            'r_binedges': r_binedges
            }

def plot_flux_est(flux_raw, flux_est, t_binedges, r_binedges):
    vmin = np.nanmin(flux_raw[flux_raw > 0]) / 1e6
    # mask_inf_dc = np.isfinite(flux_est) & (flux_est <= 40e16)  # mask to remove infinite values and anything too large
    mask_inf_dc = np.isfinite(flux_est)  # mask to remove infinite values and anything too large
    vmax = np.nanmax(flux_est[mask_inf_dc]) / 1e6

    fig = plt.figure(dpi=400,
                     figsize=(8, 6),
                     constrained_layout=True
                     )
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    __ = ax1.pcolormesh(t_binedges,
                        r_binedges / 1e3,
                        flux_raw / 1e6,
                        cmap='viridis',
                        norm=LogNorm(vmin=vmin,
                                     vmax=vmax
                                     )
                        )
    mesh2 = ax2.pcolormesh(t_binedges,
                           r_binedges / 1e3,
                           flux_est / 1e6,
                           cmap='viridis',
                           norm=LogNorm(vmin=vmin,
                                        vmax=vmax
                                        )
                           )
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Range [km]')
    ax1.set_title('Raw')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Bin Correction')
    ax2.tick_params(labelleft=False)
    cbar = fig.colorbar(mesh2, ax=[ax1, ax2],
                        location='right',
                        pad=0.15)
    cbar.set_label('Flux [MHz]')
    [plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right') for ax in [ax1, ax2]]
    plt.show()

def optimize(Y, Z, t, Nshots, num_steps, degree, deadtime, learning_rate, rel_step_lim, max_epochs, term_persist):
    dr_t = torch.diff(t)[0]  # [s] range resolution in time

    model = DataOptimizer(degree=degree, t=t)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 0
    rel_step = 1e3 * rel_step_lim

    rel_step_lst = []
    fit_loss_lst = []
    while (rel_step > rel_step_lim) and (epoch < max_epochs):

        optimizer.zero_grad()
        lamb = model()  # (M,)

        loss = dtime_loss_fn(lamb, Z, Y, Nshots, dr_t) if (deadtime == True) else pois_loss_fn(lamb, Y, Nshots, dr_t)
        fit_loss_lst += [loss.item()]

        if epoch % 100 == 0:
            print(f"step {epoch}, loss = {loss.item():.4f}")

        # calculate relative step as an average over the last term_persist iterations
        if epoch == 0:
            rel_step_lst += [rel_step]
        else:
            rel_step_lst += [(fit_loss_lst[-2] - fit_loss_lst[-1]) / np.abs(fit_loss_lst[-2])]
            rel_step = np.abs(np.array(rel_step_lst)[-term_persist:].mean())

        loss.backward()
        optimizer.step()

        epoch += 1

    print('Exited process at epoch {}/{}'.format(epoch, max_epochs))
    with torch.no_grad():
        lamb_out = model()

    return lamb_out.detach().cpu().numpy(), model.C.detach().cpu().numpy(), model.B.detach().cpu().numpy(), fit_loss_lst


if __name__ == '__main__':
    use_sim = True
    c = 299792458  # [m/s]

    # Add the project root directory to Python path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from processing.data_preprocessor_v2 import Preprocessor
    from processing.data_processor import Processor
    from sims.gen_sim_data import GenerateData

    if use_sim:
        config_path = Path(__file__).resolve().parent.parent / "config" / "sim_deadtime_fitting_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        gd = GenerateData(config)
        dpp = Preprocessor(config)
        lamb, r = gd.generate_data()
        histogram_results = gd.load_sim_data()

        flux_vars = data_setup(gd, dpp.deadtime_correct, histogram_results)
    else:
        config_path = Path(__file__).resolve().parent.parent / "config" / "preprocessing.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        dpp = Preprocessor(config)
        dpp.run()
        histogram_results = dpp.loader.gen_histogram()

        flux_vars = data_setup(dpp.loader, dpp.deadtime_correct, histogram_results)

    cnts_raw_fine = flux_vars['cnts_raw']
    af_hist_fine = flux_vars['af_hist']
    t_binedges_fine = flux_vars['t_binedges']
    r_binedges_fine = flux_vars['r_binedges']

    num_tbins = af_hist_fine.shape[1]
    cnts_raw = torch.from_numpy(cnts_raw_fine.sum(axis=1)).float()
    af_hist = torch.from_numpy(af_hist_fine.sum(axis=1)).float() / num_tbins
    r_binedges = torch.from_numpy(r_binedges_fine).float()

    flux_raw_fine = cnts_raw_fine / np.diff(t_binedges_fine)[0] / 14.3e3 / (np.diff(r_binedges_fine)[0]/c*2)

    rep_rate = 14.3e3  # [Hz]
    t_range = t_binedges_fine[-1] - t_binedges_fine[0]  # [s]
    Nshots = t_range * rep_rate
    r_binsize = torch.diff(r_binedges)[0]  # [m] range bin size in meters
    r_binsize_t = r_binsize / c * 2  # [s] range bin size in seconds
    r_centers = r_binedges[:-1] + r_binsize / 2
    r_centers_t = r_centers / c * 2  # [s] convert range to time for optimization

    degree = 28
    num_steps = 2000
    lr=1e-1  # Learning rate
    rel_step_lim = 1e-8
    max_epochs = 10000
    term_persist = 20

    lamb_out_dead, model_C_dead, model_B_dead, loss_list_dead = optimize(Y=cnts_raw, Z=af_hist, t=r_centers_t, Nshots=Nshots,
                                                         num_steps=num_steps, degree=degree, deadtime=True,
                                                         learning_rate=lr, rel_step_lim=rel_step_lim,
                                                         max_epochs=max_epochs, term_persist=term_persist)
    lamb_out_pois, model_C_pois, model_B_pois, loss_list_pois = optimize(Y=cnts_raw, Z=af_hist, t=r_centers_t, Nshots=Nshots,
                                                         num_steps=num_steps, degree=degree, deadtime=False,
                                                         learning_rate=lr, rel_step_lim=rel_step_lim,
                                                         max_epochs=max_epochs, term_persist=term_persist)
    print('Background term: deadtime {:.0f} Hz, poisson {:.0f} Hz'.format(model_B_dead[0], model_B_pois[0]))

    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    ax.plot(cnts_raw/r_binsize_t/Nshots/1e6, r_centers/1e3, 'o', alpha=0.5, label='Raw')
    ax.plot(lamb_out_pois/1e6, r_centers/1e3, '-', alpha=0.7, label='Poisson Fit')
    ax.plot(lamb_out_dead/1e6, r_centers/1e3, '-', alpha=0.7, label='Deadtime Fit')
    if use_sim:
        ax.plot(lamb/1e6, r/1e3, '-', alpha=0.7, label='Simulated Truth')
        ax.set_ylim([gd.r_plot_min, gd.r_plot_max])
    # ax.set_xlim([0, 250])
    ax.set_xlabel('Flux [MHz]')
    ax.set_ylabel('Range [km]')
    ax.set_title('Fit: Degree {}'.format(degree))
    # ax.set_xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    ax.plot(range(len(loss_list_pois)), loss_list_pois, label='Poisson')
    ax.plot(range(len(loss_list_dead)), loss_list_dead, label='Deadtime')
    ax.set_title('Loss Values')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

