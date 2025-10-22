import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import RectangleSelector

# TODO: create non-fractional binning mode in addition to fractional binning


class DeadtimeCorrect:
    def __init__(self, config):
        self.deadtime = None
        self.deadtime_trim_idx = None
        self.af_bg = False
        self.rbinsize_bg_est = 50  # [m]
        self.tbinsize_bg_est = 5  # [s]

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
        residual_idx = 1 if ((loader.deadtime * self.c / 2) > loader.rbinsize) else 0

        flux_bg_sub_hg = fluxes_bg_sub_hg['flux_bg_sub'][residual_idx:]  # [Hz]
        flux_m_bg_sub_hg = fluxes_bg_sub_hg['flux_m_bg_sub'][residual_idx:]  # [Hz]
        flux_dc_bg_sub_hg = fluxes_bg_sub_hg['flux_dc_bg_sub'][residual_idx:]  # [Hz]
        r_binedges_m_hg = fluxes_bg_sub_hg['r_binedges_m'][residual_idx:]  # [m]
        r_binedges_dc_hg = fluxes_bg_sub_hg['r_binedges_dc'][residual_idx:]  # [m]
        t_binedges_m_hg = fluxes_bg_sub_hg['t_binedges_m']  # [s]
        t_binedges_dc_hg = fluxes_bg_sub_hg['t_binedges_dc']  # [s]

        flux_bg_sub_lg = fluxes_bg_sub_lg['flux_bg_sub']  # [Hz]
        flux_m_bg_sub_lg = fluxes_bg_sub_lg['flux_m_bg_sub']  # [Hz]
        flux_dc_bg_sub_lg = fluxes_bg_sub_lg['flux_dc_bg_sub']  # [Hz]
        # r_binedges_m_lg = fluxes_bg_sub_lg['r_binedges_m']  # [m]
        # t_binedges_m_lg = fluxes_bg_sub_lg['t_binedges_m']  # [s]
        # r_binedges_dc_lg = fluxes_bg_sub_lg['r_binedges_dc']  # [m]
        # t_binedges_dc_lg = fluxes_bg_sub_lg['t_binedges_dc']  # [s]

        diff_olap_flux = flux_bg_sub_hg / flux_bg_sub_lg
        diff_olap_flux_m = flux_m_bg_sub_hg / flux_m_bg_sub_lg
        diff_olap_flux_dc = flux_dc_bg_sub_hg / flux_dc_bg_sub_lg

        diff_olap_flux_masked = np.ma.masked_where((diff_olap_flux < 0), diff_olap_flux)
        diff_olap_flux_m_masked = np.ma.masked_where((diff_olap_flux_m < 0), diff_olap_flux_m)
        diff_olap_flux_dc_masked = np.ma.masked_where((diff_olap_flux_dc < 0), diff_olap_flux_dc)

        all_diff_olap_vals = np.ma.concatenate([diff_olap_flux_masked,
                                                diff_olap_flux_m_masked,
                                                diff_olap_flux_dc_masked]).compressed()
        vmin = np.nanmin(all_diff_olap_vals)
        # vmax = np.nanmax(all_diff_olap_vals)
        vmax = 600

        if diff_olap_flux.shape[1] == 1:
            rbinsize = r_binedges_dc_hg[1] - r_binedges_dc_hg[0]  # [m]
            r_centers = r_binedges_dc_hg[:-1] + (rbinsize / 2)  # [m]
            fig = plt.figure(dpi=400,
                             figsize=(10, 5))
            ax = fig.add_subplot(111)
            ax.plot(r_centers / 1e3, diff_olap_flux_masked, label='No Correction')
            ax.plot(r_centers / 1e3, diff_olap_flux_m_masked, label='Mueller')
            ax.plot(r_centers / 1e3, diff_olap_flux_dc_masked, label='Deadtime Model')
            ax.set_xlabel('Range [km]')
            ax.set_ylabel('Differential Overlap (HG/LG)')
            ax.set_yscale('log')
            plt.legend()
            plt.tight_layout()
            plt.show()

        fig = plt.figure(dpi=400,
                         figsize=(10, 5),
                         constrained_layout=True
                         )
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        mesh1 = ax1.pcolormesh(t_binedges_dc_hg,
                               r_binedges_dc_hg / 1e3,
                               diff_olap_flux_masked,
                               cmap='viridis'
                               )
        __ = ax2.pcolormesh(t_binedges_m_hg,
                            r_binedges_m_hg / 1e3,
                            diff_olap_flux_m_masked,
                            cmap='viridis'
                            )
        mesh3 = ax3.pcolormesh(t_binedges_dc_hg,
                               r_binedges_dc_hg / 1e3,
                               diff_olap_flux_dc_masked,
                               cmap='viridis'
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
        mesh1 = ax1.pcolormesh(t_binedges_dc_hg,
                               r_binedges_dc_hg / 1e3,
                               diff_olap_flux_masked,
                               cmap='viridis',
                               norm=LogNorm(vmin=vmin,
                                            vmax=vmax
                                            )
                               )
        __ = ax2.pcolormesh(t_binedges_m_hg,
                            r_binedges_m_hg / 1e3,
                            diff_olap_flux_m_masked,
                            cmap='viridis',
                            norm=LogNorm(vmin=vmin,
                                         vmax=vmax
                                         )
                            )
        mesh3 = ax3.pcolormesh(t_binedges_dc_hg,
                               r_binedges_dc_hg / 1e3,
                               diff_olap_flux_dc_masked,
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

        quit()

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
        loader.tbinsize, loader.rbinsize = self.tbinsize_bg_est, self.rbinsize_bg_est  # [s], [m] Large bin sizes
        loader.xlim = [max(0, loader.xlim[0] - (loader.tbinsize * 5)), loader.xlim[1] + (loader.tbinsize * 5)]  # [s]
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
        print('Raw flux background estimate: {:.3e} Hz'.format(bg_flux_raw))

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

    def calc_af_hist(self, histogram_results, loader):
        """
        Method to calculate active-fraction histogram using fractional binning.
        """

        start_time = time.time()

        # Load relevant chunks
        # loader.load_chunk()
        dr_hist = loader.rbinsize  # [m] histogram range resolution
        dt_hist = loader.tbinsize  # [s] histogram time resolution

        # Load bin edges. Remember, min_range_dtime is min_range minus one deadtime interval
        hist_t_binedges = histogram_results['t_binedges']
        hist_r_binedges = histogram_results['r_binedges']
        min_range_dtime, max_range = hist_r_binedges[0], hist_r_binedges[-1]

        # Fine-resolution bin size used to calculate active-fraction (AF) fractional binning
        dr_af = (self.pulse_width * self.c / 2) if loader.fractional_bin else loader.rbinsize  # [m]
        dt_af = 1 / self.PRF  # [s]
        deadtime = self.deadtime_lg if loader.low_gain else self.deadtime_hg  # [s]

        # Create fine-res range array
        rvals_af = np.arange(min_range_dtime + (dr_af / 2), max_range, dr_af)  # [m]
        n_rbins_af = len(rvals_af)

        # Number of fine-res bins per coarse (histogram) bin
        shots_per_hist_bin = round(dt_hist / dt_af)
        af_rbins_per_hist_bin = round(dr_hist / dr_af)
        deadtime_nbins = np.floor(deadtime / (dr_af / self.c * 2)).astype(int)  # fine-res bin number in deadtime

        # Unpack time-tag ranges and shots
        ranges = np.concatenate([da.values.ravel() for da in loader.ranges_tot])  # [m]
        shots_time = np.concatenate([da.values.ravel() for da in loader.shots_time_tot])  # [s]

        # Limit shots and ranges to within xlim (time) window
        start_idx = np.argmin(np.abs(shots_time - (hist_t_binedges[0])))
        end_idx = np.argmin(np.abs(shots_time - (hist_t_binedges[-1])))
        shots_use = shots_time[start_idx:end_idx]
        ranges_use = ranges[start_idx:end_idx]

        # Limit shots and ranges to within ylim (range) window
        range_condition = (ranges_use <= max_range) & (ranges_use >= min_range_dtime)
        shots_use = shots_use[range_condition]
        ranges_use = ranges_use[range_condition]

        # Create nested list of unique shot indices to iterate for AF calculation
        shots_vals = np.unique(shots_use)
        num_dt = round((shots_vals[-1] - shots_vals[0]) / dt_hist)
        shot_indices = [np.where(shots_use == val)[0] for val in shots_vals]

        # Organize nested list of shot indices based on histogram column
        shot_indices_split = []
        prior_cutoff_idx = 0
        for shot in np.arange(shots_vals[0] + dt_hist, shots_vals[0] + (num_dt + 1) * dt_hist, dt_hist):
            shot_cutoff_idx = np.searchsorted(shots_vals, shot, side='right')
            shot_indices_split.append(shot_indices[prior_cutoff_idx:shot_cutoff_idx])
            prior_cutoff_idx = shot_cutoff_idx
        shot_indices_flat = [np.concatenate(group) for group in shot_indices_split if group]  # Flatten each column
        tbin_num = len(shot_indices_flat)  # Number of histogram time bins

        # Fine-res range bin index to trim up to so range array can integer downsample
        trim_range_idx = round(((len(rvals_af) * dr_af) - (len(rvals_af) * dr_af) % dr_hist) / dr_af)
        rbin_num_trim = round(trim_range_idx /
                              af_rbins_per_hist_bin)  # Histogram range bins number to trim to for factorizability

        # Loop through each histogram time column and subtract range bins where the detector was inactive
        af_hist = np.zeros((rbin_num_trim, tbin_num))
        af = np.ones((trim_range_idx, tbin_num)) * shots_per_hist_bin
        for i, idx in enumerate(shot_indices_flat):
            # Shot-specific ranges
            ranges_shot = ranges_use[idx]

            # Locate start and end indices for deadtime inactivity
            deadtime_start_idx = np.searchsorted(rvals_af, ranges_shot, side='left')
            deadtime_end_idx = np.clip(deadtime_start_idx + deadtime_nbins, a_min=0, a_max=n_rbins_af)

            # Mark dead regions by subtracting 1
            for start, end in zip(deadtime_start_idx, deadtime_end_idx):
                af[start:end, i] -= 1

            # Average to match histogram range bins
            af_trim = af[:, i]
            af_reshaped = af_trim.reshape(-1, af_rbins_per_hist_bin)
            af_hist[:, i] = af_reshaped.mean(axis=1)

        # Normalize AF histogram
        self.deadtime_trim_idx = round(deadtime_nbins / af_rbins_per_hist_bin)
        af_hist = af_hist[self.deadtime_trim_idx:, :] / shots_per_hist_bin

        print('Elapsed time: {} s'.format(time.time() - start_time))

        # Plot AF histogram
        if not self.af_bg:
            extent_t0, extent_t1 = (shots_vals[0] - (dt_hist / 2)), (shots_vals[-1] + (dt_hist / 2))  # [s]
            extent_r0, extent_r1 = (hist_r_binedges[round(deadtime_nbins / af_rbins_per_hist_bin)] / 1e3), \
                                   (hist_r_binedges[round((trim_range_idx - 1) / af_rbins_per_hist_bin)] / 1e3)  # [m]
            fig = plt.figure(figsize=(8, 4),
                             dpi=400
                             )
            ax = fig.add_subplot(111)
            im = ax.imshow(af_hist,
                           aspect='auto',
                           origin='lower',
                           cmap='viridis',
                           extent=[extent_t0, extent_t1, extent_r0, extent_r1]
                           )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('AF Value')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Range [km]')
            plt.show()

        return {
            'af_hist': af_hist,
            'rbin_num_trim': rbin_num_trim
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
        rbin_num_trim = af_results['rbin_num_trim']

        flux_raw = flux_raw[self.deadtime_trim_idx:rbin_num_trim]  # [Hz]
        r_binedges = r_binedges[self.deadtime_trim_idx:(rbin_num_trim + 1)]  # [m]

        flux_est = flux_raw / af_hist

        # If single profile, then plot it.
        if af_hist.shape[1] == 1:
            fig = plt.figure(dpi=400,
                             figsize=(10, 5)
                             )
            ax = fig.add_subplot(111)
            rcenters = r_binedges[:-1] - (r_binedges[1] - r_binedges[0]) / 2  # [m]
            ax.plot(rcenters / 1e3, flux_raw / 1e6, label='Measured Flux', alpha=0.75)
            ax.plot(rcenters / 1e3, flux_est / 1e6, label='Corrected Flux', alpha=0.75)
            ax.set_xlabel('Range [km]')
            ax.set_ylabel('Flux [MHz]')
            ax.set_title('Deadtime Corrected Flux (1D)')
            ax.set_yscale('log')
            plt.legend()
            plt.tight_layout()
            plt.show()

        return {
            'flux_dc_est': flux_est,
            'flux_raw': flux_raw,
            'r_binedges': r_binedges,
            't_binedges': t_binedges,
            'rbin_num_trim': rbin_num_trim
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
        rbin_num_trim = dc_results['rbin_num_trim']
        flux_dc = dc_results['flux_dc_est']  # [Hz]
        flux_raw = dc_results['flux_raw']  # [Hz]

        bg_flux_raw = deadtime_bg_results['bg_flux_raw']  # [Hz]
        bg_flux_mueller = deadtime_bg_results['bg_flux_mueller']  # [Hz]

        flux_m = flux_m[self.deadtime_trim_idx:rbin_num_trim]  # [Hz]
        r_binedges_m = r_binedges_m[self.deadtime_trim_idx:(rbin_num_trim + 1)]  # [m]

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

