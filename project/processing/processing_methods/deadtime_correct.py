import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import RectangleSelector
from scipy.signal import fftconvolve
from scipy.optimize import least_squares

# TODO: create non-fractional binning mode in addition to fractional binning


class DeadtimeCorrect:
    def __init__(self, config):
        self.deadtime = None
        self.deadtime_trim_idx = None
        self.af_bg = False
        self.rbinsize_bg_est = 50  # [m]

        # Constants
        self.c = config['constants']['c']  # [m/s] speed of light

        # System Params
        self.PRF = config['system_params']['PRF']  # [Hz] laser repetition rate
        self.unwrap_modulo = config['system_params']['unwrap_modulo']  # clock rollover count
        self.clock_res = config['system_params']['clock_res']  # [s] clock resolution
        self.deadtime_hg = config['system_params']['deadtime_hg']  # [s] high-gain detector deadtime
        self.deadtime_lg = config['system_params']['deadtime_lg']  # [s] low-gain detector deadtime
        self.pulse_width = config['system_params']['pulse_width']  # [s] FWHM

        # Process Params
        self.apply_corrections = config['process_params']['apply_corrections']  # TRUE value applies Mueller Correction
        self.active_fraction = config['process_params']['active_fraction']  # TRUE value applies active-fraction calculation

        # Plot params
        self.rbinsize = config['plot_params']['rbinsize']  # [m] range bin size
        self.tbinsize = config['plot_params']['tbinsize']  # [s] time bin size


    def plot_diff_overlap(self, fluxes_bg_sub_hg, fluxes_bg_sub_lg, loader):
        flux_bg_sub_hg = fluxes_bg_sub_hg['flux_bg_sub']  # [Hz]
        flux_m_bg_sub_hg = fluxes_bg_sub_hg['flux_m_bg_sub']  # [Hz]
        flux_dc_bg_sub_hg = fluxes_bg_sub_hg['flux_dc_bg_sub']  # [Hz]
        r_binedges = fluxes_bg_sub_hg['r_binedges_dc']  # [m]
        t_binedges = fluxes_bg_sub_hg['t_binedges_dc']  # [s]

        flux_bg_sub_lg = fluxes_bg_sub_lg['flux_bg_sub']  # [Hz]
        flux_m_bg_sub_lg = fluxes_bg_sub_lg['flux_m_bg_sub']  # [Hz]
        flux_dc_bg_sub_lg = fluxes_bg_sub_lg['flux_dc_bg_sub']  # [Hz]

        d_olap = flux_bg_sub_hg / flux_bg_sub_lg
        d_olap_m = flux_m_bg_sub_hg / flux_m_bg_sub_lg
        d_olap_dc = flux_dc_bg_sub_hg / flux_dc_bg_sub_lg

        d_olap = np.ma.masked_where((d_olap < 0), d_olap)
        d_olap_m = np.ma.masked_where((d_olap_m < 0), d_olap_m)
        d_olap_dc = np.ma.masked_where((d_olap_dc < 0), d_olap_dc)

        all_diff_olap_vals = np.ma.concatenate([d_olap,
                                                d_olap_m,
                                                d_olap_dc]).compressed()
        vmin = np.nanmin(all_diff_olap_vals)
        vmax = np.nanmax(all_diff_olap_vals)
        # vmax = 600

        rbinsize = r_binedges[1] - r_binedges[0]  # [m]
        r_centers = r_binedges[:-1] + (rbinsize / 2)  # [m]
        fig = plt.figure(dpi=400,
                         figsize=(10, 5)
                         )
        ax = fig.add_subplot(111)
        colors = ['C0', 'C1', 'C2']
        for i in range(len(d_olap[0, :])):
            ax.plot(d_olap[:, i], r_centers / 1e3, '-', color=colors[0], alpha=0.25)
            ax.plot(d_olap_m[:, i], r_centers / 1e3, '-', color=colors[1], alpha=0.25)
            ax.plot(d_olap_dc[:, i], r_centers / 1e3, '-', color=colors[2], alpha=0.25)
        ax.set_ylabel('Range [km]')
        ax.set_xlabel('Differential Overlap (HG/LG)')
        ax.set_title('Differential Overlap')
        ax.set_xscale('log')
        handles = [
            plt.Line2D([0], [0], color=colors[0], linestyle='-', label='No Correction'),
            plt.Line2D([0], [0], color=colors[1], linestyle='-', label='Mueller'),
            plt.Line2D([0], [0], color=colors[2], linestyle='-', label='Deadtime Model'),
        ]
        plt.legend(handles=handles)
        plt.tight_layout()
        plt.show()

        fig = plt.figure(dpi=400,
                         figsize=(10, 5),
                         constrained_layout=True
                         )
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        __ = ax1.pcolormesh(t_binedges,
                               r_binedges / 1e3,
                               d_olap,
                               cmap='viridis',
                               vmin=vmin,
                               vmax=vmax
                               )
        __ = ax2.pcolormesh(t_binedges,
                            r_binedges / 1e3,
                            d_olap_m,
                            cmap='viridis',
                            vmin=vmin,
                            vmax=vmax
                            )
        mesh3 = ax3.pcolormesh(t_binedges,
                               r_binedges / 1e3,
                               d_olap_dc,
                               cmap='viridis',
                               vmin=vmin,
                               vmax=vmax
                               )
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Range [km]')
        ax1.set_title('Raw')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Mueller')
        ax2.tick_params(labelleft=False)
        ax3.set_xlabel('Time [s]')
        ax3.set_title('Deadtime Model')
        ax3.tick_params(labelleft=False)
        cbar = fig.colorbar(mesh3, ax=[ax1, ax2, ax3],
                            location='right',
                            pad=0.15)
        cbar.set_label('Differential Overlap (HG/LG)')
        [plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right') for ax in [ax1, ax2, ax3]]
        plt.show()

        fig = plt.figure(dpi=400,
                         figsize=(10, 5),
                         constrained_layout=True
                         )
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        __ = ax1.pcolormesh(t_binedges,
                               r_binedges / 1e3,
                               d_olap,
                               cmap='viridis',
                               norm=LogNorm(vmin=vmin,
                                            vmax=vmax
                                            )
                               )
        __ = ax2.pcolormesh(t_binedges,
                            r_binedges / 1e3,
                            d_olap_m,
                            cmap='viridis',
                            norm=LogNorm(vmin=vmin,
                                         vmax=vmax
                                         )
                            )
        mesh3 = ax3.pcolormesh(t_binedges,
                               r_binedges / 1e3,
                               d_olap_dc,
                               cmap='viridis',
                               norm=LogNorm(vmin=vmin,
                                            vmax=vmax
                                            )
                               )
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Range [km]')
        ax1.set_title('Raw')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Mueller')
        ax2.tick_params(labelleft=False)
        ax3.set_xlabel('Time [s]')
        ax3.set_title('Deadtime Model')
        ax3.tick_params(labelleft=False)
        cbar = fig.colorbar(mesh3, ax=[ax1, ax2, ax3],
                            location='right',
                            pad=0.15)
        cbar.set_label('Differential Overlap (HG/LG)')
        [plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right') for ax in [ax1, ax2, ax3]]
        plt.show()

        return {
            'd_olap': d_olap,
            'd_olap_m': d_olap_m,
            'd_olap_dc': d_olap_dc,
            'r_binedges': r_binedges,
            't_binedges': t_binedges
        }

    def deadtime_bg_calc(self, loader, plotter):
        """
        Method to estimate background flux.
        """

        # Store config value to restore later
        load_xlim, load_ylim = loader.load_xlim, loader.load_ylim
        plot_xlim, plot_ylim = plotter.plot_xlim, plotter.plot_ylim
        tbinsize, rbinsize = loader.tbinsize, loader.rbinsize

        # When estimating background, load entire vertical range while limiting time bounds. This balances load time.
        loader.load_xlim, loader.load_ylim = True, False
        plotter.plot_xlim, plotter.plot_ylim = False, False

        x0, x1 = loader.xlim
        dt = loader.tbinsize
        width = x1 - x0

        if dt < 5:
            loader.xlim = [max(0, x0 - dt * 3), x1 + dt * 3]
            tbinsize_bg_est = (loader.xlim[1] - loader.xlim[0]) / 5
        elif dt > 50:
            mid = (x0 + x1) / 2
            loader.xlim, tbinsize_bg_est = [mid - 25, mid + 25], 10
        else:
            tbinsize_bg_est = width / 5

        loader.tbinsize, loader.rbinsize = tbinsize_bg_est, self.rbinsize_bg_est  # [s], [m] Large bin sizes
        loader.gen_hist_bg = True

        # Load histogram that includes background signal
        histogram_results_bg_est = loader.gen_histogram()
        flux_raw_bg_est = histogram_results_bg_est['flux_raw']
        r_binedges_bg_est = histogram_results_bg_est['r_binedges']
        t_binedges_bg_est = histogram_results_bg_est['t_binedges']

        # Select window to estimate background using box-selection GUI or typing in values.
        while True:
            show_hg = input('\nShow histogram to estimate background? (Y/N)')

            # Rectangle selection GUI for window
            if (show_hg == 'Y') or (show_hg == 'y'):
                selected_region = self.plot_bg_est(flux_raw_bg_est, t_binedges_bg_est, r_binedges_bg_est, loader)

                min_r_bg, max_r_bg = selected_region['y0'], selected_region['y1']  # [km]
                min_t_bg, max_t_bg = selected_region['x0'], selected_region['x1']  # [s]
                break
            # Typing window coordinates manually
            elif (show_hg == 'N') or (show_hg == 'n'):
                bg_ranges = input('Please input range bounds [km] for background estimation (comma separated):')
                bg_times = input('Please input time bounds [s] for background estimation (comma separated):')

                min_r_bg, max_r_bg = map(float, bg_ranges.split(","))  # [km]
                min_t_bg, max_t_bg = map(float, bg_times.split(","))  # [s]
                break
            else:
                print('Please input "Y" or "N".')

        bg_r_edges = np.array([min_r_bg, max_r_bg]) * 1e3  # [m]
        bg_t_edges = np.array([min_t_bg, max_t_bg])  # [s]

        # Calculate background from raw data
        bg_flux_raw = loader.calc_bg(flux_raw_bg_est,
                                     r_binedges_bg_est,
                                     t_binedges_bg_est,
                                     bg_r_edges,
                                     bg_t_edges)  # [Hz]
        if loader.bg_sub:
            print('Raw flux background estimate: {:.3e} Hz'.format(bg_flux_raw))
        else:
            print('Skipping background estimate. Set background to {} Hz.'.format(bg_flux_raw))

        # Apply Mueller correction and then calculate background
        mueller_results_bg_est = self.mueller_correct(histogram_results_bg_est, loader)  # [Hz]
        flux_mueller_bg_est = mueller_results_bg_est['flux_mueller']  # [Hz]
        r_binedges_m_bg_est = mueller_results_bg_est['r_binedges']  # [m]
        t_binedges_m_bg_est = mueller_results_bg_est['t_binedges']  # [s]
        bg_flux_mueller = loader.calc_bg(flux_mueller_bg_est,
                                         r_binedges_m_bg_est,
                                         t_binedges_m_bg_est,
                                         bg_r_edges,
                                         bg_t_edges)  # [Hz]
        if loader.bg_sub:
            print('Mueller-corrected flux background estimate: {:.3e} Hz'.format(bg_flux_mueller))

        # Restore config file values
        loader.load_xlim, loader.load_ylim = load_xlim, load_ylim
        plotter.plot_xlim, plotter.plot_ylim = plot_xlim, plot_ylim
        loader.tbinsize, loader.rbinsize = tbinsize, rbinsize

        return {
            'bg_flux_raw': bg_flux_raw,
            'bg_flux_mueller': bg_flux_mueller
        }

    def mueller_correct(self, histogram_results, loader):
        """
        Apply the Mueller correction to flux histogram.
        """

        flux_raw = histogram_results['flux_raw']
        r_binedges = histogram_results['r_binedges']
        t_binedges = histogram_results['t_binedges']
        self.deadtime = self.deadtime_lg if loader.low_gain else self.deadtime_hg

        # Calculate flux estimate based on Mueller Correction
        flux_mueller = flux_raw / (1 - self.deadtime * flux_raw)  # [Hz]

        return {
            'flux_mueller': flux_mueller,
            'r_binedges': r_binedges,
            't_binedges': t_binedges
        }

    def measure_deadtime(self, ranges, loader):
        """
        Empirically measure and plot the deadtime from time tags from saturating measurement.

        Inputs:
            ranges (nx1): [m] range array
        """

        # Calculate inter-arrival flight times
        flight_times = ranges / self.c * 2  # [s]
        intr_times = np.diff(flight_times)  # [s] Inter-arrival times

        start = time.time()
        max_val = 100e-9  # [s] capping range to explore deadtime value from inter-arrival times. Usually around 20-40
        # ns for SPCM detectors.
        bins = np.arange(0, max_val, self.clock_res)  # [s]
        cnts, bin_edges = np.histogram(intr_times, bins)  # [], [s]
        print('Time elapsed {} s'.format(time.time() - start))

        # Deadtime estimate is the inter-arrival histogram peak location
        binsize = bin_edges[1] - bin_edges[0]  # [s]
        bin_centers = bin_edges[:-1] + 0.5 * binsize  # [s]
        bin_widths = np.diff(bin_edges)  # [s]
        deadtime_estimate = bin_centers[np.argmax(cnts)]  # [s]
        print('Deadtime Estimate: {:.4f} ns'.format(deadtime_estimate * 1e9))
        setattr(self, f"deadtime_{'lg' if loader.low_gain else 'hg'}", deadtime_estimate)

        # Plot inter-arrival time histogram
        xmax = deadtime_estimate * 1e9  # [ns]
        ymax = np.max(cnts)
        fig = plt.figure(dpi=400,
                         figsize=(4, 3)
                         )
        ax = fig.add_subplot(111)
        ax.bar(bin_centers * 1e9, cnts,
               width=bin_widths*1e9,
               align="edge",
               edgecolor="black"
               )
        ax.annotate("Deadtime Estimate {:.1f} ns".format(deadtime_estimate*1e9),
                    xy=(xmax, ymax),
                    xytext=(xmax, 1.3 * ymax),
                    ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="->")
                    )
        ax.set_xlabel("$\Delta t$ [ns]")
        ax.set_ylabel("Counts")
        ax.set_title('Inter-Arrival Times {} Channel\n{}'.format("Low-Gain" if loader.low_gain is True else "High-Gain", loader.timestamp))
        ax.set_ylim(0, 1.5 * ymax)
        plt.tight_layout()
        plt.show()

    def calc_af_hist_convolution(self, histogram_results, loader):
        """
        Method to calculate active-fraction histogram using fractional binning.
        """

        start_time = time.time()

        # Load bin edges. Remember, min_range_dtime is min_range minus one deadtime interval
        hist_t_binedges = histogram_results['t_binedges']
        hist_r_binedges = histogram_results['r_binedges']
        H = histogram_results['cnts_raw']
        rbinsize = loader.rbinsize
        tbinsize = loader.tbinsize
        PRF = loader.PRF

        # min_range_dtime, max_range = hist_r_binedges[0], hist_r_binedges[-1]
        nrbins = len(hist_r_binedges[:-1])

        deadtime_range = loader.deadtime * self.c / 2
        K = np.floor(deadtime_range / rbinsize).astype(int)  # number of bins (floor round) that occupy deadtime
        dtime_kern = np.ones(K)
        # Handle remainder. If deadtime is longer than binsize, then tack on fractional bin.
        # If deadtime is shorter, then only include single fractional bin.
        if deadtime_range > rbinsize:
            remainder = deadtime_range % rbinsize
        else:
            remainder = deadtime_range / rbinsize
        dtime_kern = np.append(dtime_kern, remainder)

        N = tbinsize * PRF  # number of shots per time bin
        d = fftconvolve(H, dtime_kern[:, None], mode='full') / N
        d = d[:nrbins, :]
        a = 1 - d

        # Normalize AF histogram
        af_hist = a[K:, :]
        self.deadtime_trim_idx = K

        print('Elapsed time: {} s'.format(time.time() - start_time))

        # Plot AF histogram
        if not self.af_bg:
            extent_t0, extent_t1 = (hist_t_binedges[0] - (tbinsize / 2)), \
                                   (hist_t_binedges[-1] + (tbinsize / 2))  # [s]
            extent_r0, extent_r1 = (hist_r_binedges[K] / 1e3), \
                                   (hist_r_binedges[-1] / 1e3)  # [m]
            fig = plt.figure(figsize=(8, 4),
                             dpi=400
                             )
            ax = fig.add_subplot(111)
            im = ax.imshow(af_hist,
                           aspect='auto',
                           origin='lower',
                           cmap='viridis',
                           extent=[extent_t0,
                                   extent_t1,
                                   extent_r0,
                                   extent_r1
                                   ]
                           )
            cbar = fig.colorbar(im,
                                ax=ax
                                )
            cbar.set_label('AF Value')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Range [km]')
            plt.show()

        return {
            'af_hist': af_hist
        }

    def plot_bg_est(self, flux, t_binedges, r_binedges, loader):
        onselect, selected_region = self.selector_callback()

        fig = plt.figure(dpi=400,
                         figsize=(10, 5))
        ax = fig.add_subplot(111)
        mesh = ax.pcolormesh(t_binedges,
                             r_binedges / 1e3,
                             flux,
                             cmap='viridis',
                             norm=LogNorm(vmin=np.nanmin(flux[flux > 0]),
                                          vmax=np.nanmax(flux))
                             )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label('Flux [Hz]')
        ax.set_xlim([t_binedges[0], t_binedges[-1]])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Range [km]')
        ax.set_title('Scale {:.1f} m x {:.2f} s\n{} {}'.format(loader.rbinsize, loader.tbinsize,
                                                               "Low Gain" if loader.low_gain else "High Gain",
                                                               loader.timestamp))

        # Create the rectangle selector
        __ = RectangleSelector(
            ax, onselect,
            drawtype='box',
            useblit=True,
            button=[1],  # left mouse button
            minspanx=loader.tbinsize,
            minspany=(loader.rbinsize / 1e3),
            spancoords='data',
            interactive=True
        )

        plt.show(block=True)

        return selected_region

    def deadtime_model_correct(self, af_results, histogram_results):
        """
        Apply deadtime-model correction by inverting flux and active-fraction histograms.
        """

        t_binedges = histogram_results['t_binedges']  # [s]
        r_binedges = histogram_results['r_binedges']  # [m]
        flux_raw = histogram_results['flux_raw']  # [Hz]
        af_hist = af_results['af_hist']

        flux_raw = flux_raw[self.deadtime_trim_idx:]  # [Hz]
        r_binedges = r_binedges[self.deadtime_trim_idx:]  # [m]

        flux_est = flux_raw / af_hist

        # If single profile, then plot it.
        if af_hist.shape[1] == 1:
            fig = plt.figure(dpi=400,
                             figsize=(10, 5)
                             )
            ax = fig.add_subplot(111)
            rcenters = r_binedges[:-1] - (r_binedges[1] - r_binedges[0]) / 2  # [m]
            ax.plot(flux_raw / 1e6, rcenters / 1e3, label='Measured Flux', alpha=0.75)
            ax.plot(flux_est / 1e6, rcenters / 1e3, label='Corrected Flux', alpha=0.75)
            ax.set_ylabel('Range [km]')
            ax.set_xlabel('Flux [MHz]')
            ax.set_title('Deadtime Corrected Flux (1D)')
            ax.set_xscale('log')
            plt.legend()
            plt.tight_layout()
            plt.show()

        return {
            'flux_dc_est': flux_est,
            'flux_raw': flux_raw,
            'r_binedges': r_binedges,
            't_binedges': t_binedges
        }

    def plot_binwise_corrections(self, mueller_results, dc_results, deadtime_bg_results):
        """
        Plot raw data, Mueller, and deadtime-model corrections to compare.
        """

        r_binedges_m = mueller_results['r_binedges']  # [m]
        t_binedges_m = mueller_results['t_binedges']  # [s]
        flux_m = mueller_results['flux_mueller']  # [Hz]

        r_binedges_dc = dc_results['r_binedges']  # [m]
        t_binedges_dc = dc_results['t_binedges']  # [s]
        flux_dc = dc_results['flux_dc_est']  # [Hz]
        flux_raw = dc_results['flux_raw']  # [Hz]

        bg_flux_raw = deadtime_bg_results['bg_flux_raw']  # [Hz]
        bg_flux_mueller = deadtime_bg_results['bg_flux_mueller']  # [Hz]

        flux_m = flux_m[self.deadtime_trim_idx:]  # [Hz]
        r_binedges_m = r_binedges_m[self.deadtime_trim_idx:]  # [m]

        flux_raw -= bg_flux_raw  # [Hz]
        flux_m -= bg_flux_mueller  # [Hz]
        flux_dc -= bg_flux_mueller  # [Hz]

        vmin = np.nanmin(flux_raw[flux_raw > 0]) / 1e6
        vmax = max(np.nanmax(flux_dc), np.nanmax(flux_m)) / 1e6

        fig = plt.figure(dpi=400,
                         figsize=(10, 5),
                         constrained_layout=True
                         )
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        __ = ax1.pcolormesh(t_binedges_dc,
                            r_binedges_dc / 1e3,
                            flux_raw / 1e6,
                            cmap='viridis',
                            norm=LogNorm(vmin=vmin,
                                         vmax=vmax
                                         )
                            )
        __ = ax2.pcolormesh(t_binedges_m,
                            r_binedges_m / 1e3,
                            flux_m / 1e6,
                            cmap='viridis',
                            norm=LogNorm(vmin=vmin,
                                         vmax=vmax
                                         )
                            )
        mesh3 = ax3.pcolormesh(t_binedges_dc,
                               r_binedges_dc / 1e3,
                               flux_dc / 1e6,
                               cmap='viridis',
                               norm=LogNorm(vmin=vmin,
                                            vmax=vmax
                                            )
                               )
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Range [km]')
        ax1.set_title('Raw')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Mueller')
        ax2.tick_params(labelleft=False)
        ax3.set_xlabel('Time [s]')
        ax3.set_title('Deadtime Model')
        ax3.tick_params(labelleft=False)
        cbar = fig.colorbar(mesh3, ax=[ax1, ax2, ax3],
                            location='right',
                            pad=0.15)
        cbar.set_label('Flux [MHz]')
        [plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right') for ax in [ax1, ax2, ax3]]
        plt.show()

        return {
            'flux_bg_sub': flux_raw,
            'flux_m_bg_sub': flux_m,
            'flux_dc_bg_sub': flux_dc,
            'r_binedges_m': r_binedges_m,
            't_binedges_m': t_binedges_m,
            'r_binedges_dc': r_binedges_dc,
            't_binedges_dc': t_binedges_dc,
            'deadtime_trim_idx': self.deadtime_trim_idx
        }

    @staticmethod
    def selector_callback():
        selected_region = {}

        def onselect(eclick, erelease):
            """Callback for rectangle selection."""
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata

            selected_region.update({
                'x0': min(x0, x1),
                'x1': max(x0, x1),
                'y0': min(y0, y1),
                'y1': max(y0, y1)
            })

            print('Selected region: {:.2f} - {:.2f} s x {:.2f} - {:.2f} km'.format(x0, x1, y0, y1))

        return onselect, selected_region

