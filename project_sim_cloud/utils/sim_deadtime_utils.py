"""
useful functions for simulating 
and processing
photon counting with deadtime
"""

import os,sys
import numpy as np
from scipy.special import gammaln, gammainc, gammaincc

from numpy.random import default_rng
from scipy import optimize

import copy

import datetime

from typing import List, Tuple
# from tqdm.notebook import tqdm

def ensure_path(path:str):
    """
    Checks if a path exists.  If it doesn't, creates the
    necessary folders.
    path - path to check for or create
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
                print()
                print('tried creating data directory but it already exists')
                print(path)
                print()

def Num_Gradient(func,x0,step_size=1e-3):
    """
    Numerically estimate the gradient of a function at x0.
    Useful for validating analytic gradient functions.
    """
    Gradient = np.zeros((x0.size))
    for ai in range(x0.size):
        xu = x0.astype(np.float)
        xl = x0.astype(np.float)
        if x0[ai] != 0:
            xu[ai] = x0[ai]*(1+step_size)            
            xl[ai] = x0[ai]*(1-step_size)
#            Gradient[ai] = (func(xu)-func(xl))/(2*step_size)
        else:
            xu[ai] = step_size
            xl[ai] = -step_size
#            Gradient[ai] = (func(step_size)-func(-step_size))/(2*step_size)

        Gradient[ai] = (func(xu)-func(xl))/(xu[ai]-xl[ai])
    return Gradient

def Num_Gradient_Dict(func,x0,step_size=1e-3):
    """
    Numerically estimate the gradient of a function at x0 which consists
    of a dict of independent variables.
    Useful for validating analytic gradient functions.
    
    """
    
    Gradient = {}
    for var in x0.keys():
        xu = copy.deepcopy(x0)
        xl = copy.deepcopy(x0)
        if x0[var].ndim > 0:
            # handle cases where the parameter is an array
            Gradient[var] = np.zeros(x0[var].shape)
            for ai in np.ndindex(x0[var].shape):
                xu[var] = x0[var].astype(np.float)
                xl[var] = x0[var].astype(np.float)
                if x0[var][ai] != 0:
                    xu[var][ai] = x0[var][ai]*(1+step_size)            
                    xl[var][ai] = x0[var][ai]*(1-step_size)
        #            Gradient[ai] = (func(xu)-func(xl))/(2*step_size)
                else:
                    xu[var][ai] = step_size
                    xl[var][ai] = -step_size
        #            Gradient[ai] = (func(step_size)-func(-step_size))/(2*step_size)
                
                Gradient[var][ai] = (func(**xu)-func(**xl))/(xu[var][ai]-xl[var][ai])
        else:
            # handle cases where the parameter is a scalar
            xu[var] = np.float(x0[var])
            xl[var] = np.float(x0[var])
            if x0[var] != 0:
                xu[var] = x0[var]*(1+step_size)            
                xl[var] = x0[var]*(1-step_size)
    #            Gradient[ai] = (func(xu)-func(xl))/(2*step_size)
            else:
                xu[var] = step_size
                xl[var] = -step_size
    #            Gradient[ai] = (func(step_size)-func(-step_size))/(2*step_size)
            
            Gradient[var] = (func(**xu)-func(**xl))/(xu[var]-xl[var])
    return Gradient

# Simulator for deadtime data
def mc_poisson_particles(s_lst,tD,tmax,tswitch=None):
    """
    s_lst list of different discrete rates along the profile
    tD - dead time
    tmax - maximum lenght of simulation
    tswitch - list of times where the rate changes to the next
        value in s_lst
    """
    if tswitch is None:
        tswitch = [tmax]
    else:
        tswitch += [tmax]
        
    s_index = 0
    t = 0
    tdet = -1000
    ptime = []
    ctime = []
    while t < tmax:
        rate = s_lst[s_index]
        dtnew = np.random.exponential(scale=1/rate)
        t+=dtnew
        if t >= tswitch[s_index]:
            t = tswitch[s_index]
            s_index += 1
        else:
            ptime+=[t]
            if t-tdet > tD:
                ctime+=[t]
                tdet = t

    ptime = np.array(ptime)    
    ctime = np.array(ctime)
    return ptime,ctime


# Willem's simulator for photons
def int_lmbd(v_flt: float, t_arr: np.ndarray, delta_t_flt: float, lmbd_arr: np.ndarray, scaled_lmbd_arr: np.ndarray) -> float:
    """The integral of the photon rate where $\int_{0}^{v}\lambda(t')dt'$."""

    if v_flt > t_arr.max():
        err_str = 'The time stamp v is out of bound of the time span.'
        raise ValueError(err_str)

    # print(v_flt)
    # print(t_arr)
    # print(delta_t_flt)
    # print(np.ceil((t_arr - v_flt)/delta_t_flt))
    # print(np.where(np.ceil((t_arr - v_flt) / delta_t_flt) == 0)[0])
        
    # Find the time point that is closest to v
    clst_t_idx = int(np.where(np.ceil((t_arr - v_flt) / delta_t_flt) == 0)[0])
    
    # Compute the integral from 0 to t_arr[clst_t_idx]
    int_flt = scaled_lmbd_arr[0:clst_t_idx].sum()
    # Compute the intergral from t_arr[clst_t_idx] to v
    int_flt += (v_flt - t_arr[clst_t_idx]) * lmbd_arr[clst_t_idx]

    return int_flt


def inf_v_int_lmbd(s_flt: float, 
                   t_arr: np.ndarray, 
                   delta_t_flt: float,
                   lmbd_arr: np.ndarray, 
                   scaled_lmbd_arr: np.ndarray, 
                   max_int_lmbd_flt: float,
                   tol_flt: float = 1.0e-9) -> float:
    """Find smallest v such that $\int_{0}^{v}\lambda(t')dt' \geq s$."""
    
    def _root_fnc(_x: float) -> float:
        return int_lmbd(_x, t_arr, delta_t_flt, lmbd_arr, scaled_lmbd_arr) - s_flt
    
    try:
        v_flt = optimize.root_scalar(_root_fnc, bracket=[0, float(t_arr.max()) - tol_flt]).root
    except ValueError as e_obj:
        print('s_flt: ' + str(s_flt))
        print('max_int_lmbd_flt: ' + str(max_int_lmbd_flt))
        raise e_obj
    
    return v_flt


# The deadtime simulator
def photon_count_generator(t_arr: np.ndarray, 
                           lmbd_arr: np.ndarray, 
                           tau_d_flt: float = 0, 
                           tol_flt: float = 1.0e-20,
                           last_photon_flt: float=-100.0) -> Tuple[np.ndarray,np.ndarray]:
    """
    Generate non-homegenous Poisson arrival times with deadtime with Cinlar's method; see 
    Generating Nonhomogeneous Poisson Processes by Raghu Pasupathy
    (https://web.ics.purdue.edu/~pasupath/PAPERS/2011pasB.pdf).
    
    Parameters
    ----------
    t_arr: np.ndarray
        The time intervals of the photon count lambda including the last timestamp.
    lmbd_arr: np.ndarray 
        The photon rate.
    tau_d_flt: float
        The non-extended deadtime.
    tol_flt: float
        Tolerance for the `inf_v_int_lmbd` function.
    last_photon_flt: float
        pass in the last photon (modulo last bin) to wrap around
        the deadtime from the last shot.
    
    Returns
    -------
    np.ndarray
        The time instance when photons arrived. (no deadtime)
    np.ndarray
        The time instance when photons were observed. (with deadtime)
    """

    rng = default_rng()
    
    if (t_arr.ndim != 1) and (lmbd_arr.ndim != 1):
        err_str = 'The dimensions of t_arr and lmbd_arr MUST be 1.'
        raise ValueError(err_str)
    
    if t_arr.size != (lmbd_arr.size + 1):
        err_str = 'The time intervals must include the laset timestamp also.'
        raise ValueError(err_str)

    # The maximum time
    t_max_flt = float(t_arr.max())
    
    # Compute delta t
    delta_f_flt = float(np.diff(t_arr)[0])
    
    # The last time a photon was detector
    prev_dp_t_flt = last_photon_flt  # originally 0, but causes blanking at beginning of profile
        
    # The list of arrival times of photons
    ap_t_flt_lst = []  # actual photons
    ac_t_flt_lst = []  # counted photons
    
    # Compute the expected counts per interlval
    scaled_lmbd_arr = np.diff(t_arr) * lmbd_arr
    
    # Compute the maximum value of the integral
    max_int_lmbd_flt = float(np.sum(scaled_lmbd_arr))
    
    # Initialize s
    s_flt = 0
    while prev_dp_t_flt < t_max_flt:
        # Generate u
        u_flt = rng.uniform(0, 1)
        # Set s
        s_flt = s_flt - np.log(u_flt)
        
        # s must be within the range of the intergral
        if s_flt > max_int_lmbd_flt:
            break
        
        # Generate arrival time of photon
        t_flt = inf_v_int_lmbd(s_flt, 
                               t_arr, 
                               delta_f_flt,
                               lmbd_arr, 
                               scaled_lmbd_arr,
                               max_int_lmbd_flt,
                               tol_flt=tol_flt)
        
        # If the arrival time is outside the time domain, stop recording 
        # the photon arrival times
        if t_flt > t_max_flt:
            break
        
        # record the photon event
        ap_t_flt_lst.append(t_flt)

        # Check if the arrival time is outside of the deadtime
        if (t_flt - prev_dp_t_flt) > tau_d_flt:
            ac_t_flt_lst.append(t_flt) # record the detection event
            prev_dp_t_flt = t_flt
        # Else...ignore the photon since the photon detector is 
        # in deadtime
    
    return np.array(ap_t_flt_lst), np.array(ac_t_flt_lst)

def split_timetag_array(timetags,lasershot,totalshots=None):
    """
    Split the time tag data from netcdf into a list of
    arrays containing time tags for each laser shot
    inputs:
        timetags:
            array containing all the time tags
            e.g. ds['counted_timetags_chan1_run3'].values

        lasershot:
            array containing labels for what laser shot
                each timetag is associated with
            e.g. ds['counted_lasershot_chan1_run3'].values
        
        totalshots:
            total number of laser shots (provide to avoid 
                missing empty shots)
            e.g. ds['time1'].size
    returns:
        list of arrays corresponding to each laser shot
    """
    if totalshots is None:
        totalshots = lasershot.max()

    channel_photon_count_lst = []
    for laser_idx in range(totalshots):
        count_idx = np.where(lasershot==laser_idx)
        channel_photon_count_lst += [timetags[count_idx]]
    
    return channel_photon_count_lst

def split_hist_timetag_array(timetags:np.ndarray,
        lasershot:np.ndarray,laser_shot_time:np.ndarray,
        range_bins:np.ndarray,shot_time_bins:np.ndarray,
        tD:float,wrap:bool=True,n_thin:int=1,verbose:bool=False)->List[List[np.ndarray]]:
    """
    create a gridded histogram from time tags
    inputs:
        timetags - 1D array containing all time tags
        lasershot - array identifying what laser shot each time tag aligns with
        laser_shot_time - array where the the first row is the list of laser shots
            and the second column is the time associated with that shot
        range_bins - desired bin edges for histogram (in units of time)
        shot_time_bins - histogram bins for the time axis
        tD - detector dead time
        wrap - wrap deadtime effect of last profile into
            the next profile
            Defaults to True
        n_thin - number of datasets to split out (e.g. for train, validation, test sets)
        verbose - output status
    returns:
        List where each entry corresponds to a thinned set.
        Within each set there will be two np.ndarray 
            2D histogram of time tags (rows: shot_time_bin, columns: range_bins)
            2D histogram of integral time tags (rows: shot_time_bin, columns: range_bins)
            1D array containing the number of laser shots used to build those histograms
    """


    dt = np.mean(np.diff(range_bins))

    # initialize output histograms
    ret_lst = []
    for out_idx in range(n_thin):
        # first item is the standard histogram
        # second item is the active time histogram (aka integral)
        ret_lst.append([np.zeros((shot_time_bins.size-1,range_bins.size-1)),
                np.zeros((shot_time_bins.size-1,range_bins.size-1)),
                np.zeros((shot_time_bins.size-1,))])
    
    wrap_hist = None  # initialize so the first shot the detector is active
    out_idx = 0  # indicates which dataset a laser shot is going to 

    # iterate through all histogram bins
    for hist_time_idx in range(shot_time_bins.size-1):
        # locate all the laser shots that fall in the current 
        # histogram bins
        shot_sub_lst = np.where((laser_shot_time[1,:] >= shot_time_bins[hist_time_idx])& \
            (laser_shot_time[1,:] < shot_time_bins[hist_time_idx+1]))[0]
        
        for laser_idx in laser_shot_time[0,shot_sub_lst]:
            count_idx = np.where(lasershot==laser_idx)
            yint, yp, ywrap = encode_arrival_times_wrap([timetags[count_idx]],range_bins,
                              tD,dt_lam=dt,h_wrap=wrap_hist)
            out_idx = np.mod(out_idx,n_thin)

            ret_lst[out_idx][0][hist_time_idx,:]+=yp
            ret_lst[out_idx][1][hist_time_idx,:]+=yint
            ret_lst[out_idx][2][hist_time_idx]+=1  # increment the shot count

            if wrap:
                wrap_hist = ywrap
            
            out_idx+=1

        if verbose:
            print(f'completed {hist_time_idx+1} of {shot_time_bins.size-1} bins',end='\r')

    return ret_lst



def histogram_time_tags(timetags_lst,range_bins):
    """
    create a gridded histogram from time tags
    inputs:
        timetags_lst - list of arrays of time tags
        range_bins - desired bin edges for histogram
    returns:
        2D histogram of time tags (rows: laser shot, columns: range)
    """
    # set the profile bins
    hist_prof = np.zeros((len(timetags_lst),range_bins.size-1))
    for prof_index in range(len(timetags_lst)):
        hist_prof[prof_index,:]+=np.histogram(timetags_lst[prof_index],bins=range_bins)[0]
    return hist_prof

def thin_data(data,data_time,thin_frac=None,thin_num=2):
    """
    Randomly thin TCSPC data by profiles
    inputs:
        data - list of time tag data where each
            laser shot has an array of time tags
        data_time - floating point representation of the
            time of each laser shot in data
        thin_frac - list of how the data should be
            broken up in time.  Each entry correspends
            to a thinned set.  E.g.
            thin_frac = [0.1,0.4,0.5]
        thin_num - if thin_frac is not provided, the
            number of sets can be specified with this
            argument and the profiles will be split
            evenly.

    returns
        thin_data - list of list of time tag profiles
        thin_time - list of laser shot times for each profile
    """


    assert data.shape[0] == data_time.size
    
    # intialize the number of data points to draw for
    # each thinned set
    if thin_frac is None:
        # setup the draw count for each thinned set
        draw_count_list = [data_time.size//thin_num]*(thin_num-1)
    else:
        # the sum of the fractions needs to be less than 1
        assert sum(thin_frac) < 1
        
        # create a list of the number of data points in
        # each thinned set
        draw_count_list = []
        for frac in thin_frac:
            draw_count_list += [np.int(data_time.size*frac)]
            
    # Random thinning by splitting the data at its
    # base resolution
    thin_data = []
    thin_time = []
    total_index = np.arange(data_time.size)   # create list of unselected points
    for draw_count in draw_count_list:
        # create list of array positions to draw from
        draw_set = np.arange(total_index.size)
        # randomly draw from the current set of remaining indices (as array positions)
        draw_index = sorted(np.random.choice(draw_set,size=draw_count,replace=False))

        # store the index dataset indices that were drawn
        time_index = total_index[draw_index]
        # remove the indices that were drawn
        total_index = np.delete(total_index,draw_index)

        # store the dataset with time indices corresponding
        # to those drawn
        thin_data+=[data[time_index,:]]
        thin_time+=[data_time[time_index]]

    # assign all remaining points to the last set
    thin_data+=[data[total_index,:]] 
    thin_time+=[data_time[total_index]]
    return thin_data, thin_time

###   Müller functions   ###
def t_k_fnc(k_int, t_flt, tau_flt, rho_flt):
    return np.maximum(rho_flt * (t_flt - np.maximum(k_int,0) * tau_flt),0)

def p_muller(rho_flt: np.ndarray,
               k_int: np.ndarray,
               t_flt: np.ndarray,
               tau_flt: float):
    """ Calculate likelihood for Müller distribution. Assumes that rho and k and t are np.ndarrays of
    the same size and dimensions

    Parameters
    ----------
    rho: np.ndarray
        Mean photon arrival rate.
    k: np.ndarray
        Observed number of photons.
    t: np.ndarray
        Observation time interval.
    tau: float
        Dead time of detector.

    Returns
    -------
    Pm: np.ndarray
        likelihood with same dimensions as rho and k.
    """
    t_flt[t_flt==0] = 1e-20
    rho_switch_flt = k_int / ( t_flt - k_int * tau_flt )
    
    p_mul_flt = muller_upper(k_int, t_flt, tau_flt, rho_flt)
    p_mul_flt_lo = muller_lower(k_int, t_flt, tau_flt, rho_flt)
    ilow = np.nonzero( (rho_flt < rho_switch_flt) & (k_int > 3) )
    p_mul_flt[ilow] = p_mul_flt_lo[ilow]
    
    p_mul_flt = p_mul_flt /(1 + rho_flt*tau_flt)
    
    return p_mul_flt
    
def d_p_muller(rho_flt: np.ndarray,
               k_int: np.ndarray,
               t_flt: np.ndarray,
               tau_flt: float):
    """ Calculate deriviative of likelihood for Müller distribution. Assumes that rho and k and t are np.ndarrays of
    the same size and dimensions

    Parameters
    ----------
    rho: np.ndarray
        Mean photon arrival rate.
    k: np.ndarray
        Observed number of photons.
    t: np.ndarray
        Observation time interval.
    tau: float
        Dead time of detector.

    Returns
    -------
    Pm: np.ndarray
        likelihood with same dimensions as rho and k.
    """
    
    rho_switch_flt = k_int / ( t_flt - k_int * tau_flt )
    
    d_p_mul_flt = d_muller_upper(k_int, t_flt, tau_flt, rho_flt)
    d_p_mul_flt_lo = d_muller_lower(k_int, t_flt, tau_flt, rho_flt)
    ilow = np.nonzero( (rho_flt < rho_switch_flt) & ( k_int > 3 ) )
    d_p_mul_flt[ilow] = d_p_mul_flt_lo[ilow]
    
    p_mul_flt = p_muller(rho_flt, k_int, t_flt, tau_flt)
    
    d_p_mul_flt = d_p_mul_flt /(1 + rho_flt*tau_flt) - p_mul_flt * tau_flt/(1 + rho_flt*tau_flt)**2
    
    return d_p_mul_flt
    
    
def muller_upper(k_int, t_flt, tau_flt, rho_flt):
    k_max = (t_flt//tau_flt).astype(np.int)
    
    # when k < k_max
    p1 = p_upper(k_int - 1, t_flt, tau_flt, rho_flt)
    p2 = p_upper(k_int, t_flt, tau_flt, rho_flt)
    p3 = p_upper(k_int + 1, t_flt, tau_flt, rho_flt)
    
    # when k == k_max
    #     p1k = p1
    #     p2k = p2
    #     p3k = -rho_flt*t_flt + (K + 1)/(1+rho_flt*tau_flt)
    ik_int = np.nonzero(k_int == k_max)
    p3[ik_int] = -rho_flt[ik_int]*t_flt[ik_int] + (k_max[ik_int] + 1)*(1+rho_flt[ik_int]*tau_flt)
    
    # when k == k_max + 1
    #     p1kp1 = 0
    #     p2kp1 = p_lower(k_int-1, t_flt, tau_flt, rho_flt)
    #     p3kp1 = rho_flt*t_flt - K/(1+rho_flt*tau_flt)
    ik_int = np.nonzero(k_int == k_max+1)
    p1[ik_int] = 0
    p2[ik_int] = -0.5*p_upper(k_int[ik_int]-1, t_flt[ik_int], tau_flt, rho_flt[ik_int])
    p3[ik_int] = rho_flt[ik_int]*t_flt[ik_int] - k_max[ik_int]*(1+rho_flt[ik_int]*tau_flt)
#     p1 = p_upper(k_int - 1, t_flt, tau_flt, rho_flt)
#     p2 = p_upper(k_int, t_flt, tau_flt, rho_flt)
#     p3 = p_upper(k_int + 1, t_flt, tau_flt, rho_flt)
    p_mul_up_flt = p1 - 2 * p2 + p3
    return p_mul_up_flt
    
def muller_lower(k_int, t_flt, tau_flt, rho_flt):
    k_max = (t_flt//tau_flt).astype(np.int)
    
    # when k < k_max
    p1 = p_lower(k_int - 1, t_flt, tau_flt, rho_flt)
    p2 = p_lower(k_int, t_flt, tau_flt, rho_flt)
    p3 = p_lower(k_int + 1, t_flt, tau_flt, rho_flt)
    
    # when k == k_max
    #     p1k = p1
    #     p2k = p2
    #     p3k = -rho_flt*t_flt + (K + 1)/(1+rho_flt*tau_flt)
    ik_int = np.nonzero(k_int == k_max)
    p3[ik_int] = -rho_flt[ik_int]*t_flt[ik_int] + (k_max[ik_int] + 1)*(1+rho_flt[ik_int]*tau_flt)
    
    # when k == k_max + 1
    #     p1kp1 = 0
    #     p2kp1 = p_lower(k_int-1, t_flt, tau_flt, rho_flt)
    #     p3kp1 = rho_flt*t_flt - K/(1+rho_flt*tau_flt)
    ik_int = np.nonzero(k_int == k_max+1)
    p1[ik_int] = 0
    p2[ik_int] = -0.5*p_lower(k_int[ik_int]-1, t_flt[ik_int], tau_flt, rho_flt[ik_int])
    p3[ik_int] = rho_flt[ik_int]*t_flt[ik_int] - k_max[ik_int]*(1+rho_flt[ik_int]*tau_flt)
    
    p_mul_lo_flt = -1 * (p1 - 2 * p2 + p3)
    return p_mul_lo_flt

def f_k_lower(k_int, tk_flt):
    return gammainc(np.maximum(k_int + 1,1), tk_flt)*(k_int >= 0)

def f_k_upper(k_int, tk_flt):
    return gammaincc(np.maximum(k_int + 1,1), tk_flt)*(k_int >= 0)

def p_lower(k_int, t_flt, tau_flt, rho_flt):
    Tk = t_k_fnc(k_int, t_flt, tau_flt, rho_flt)
    p_flt = k_int * f_k_lower(k_int - 1, Tk) - Tk * f_k_lower(k_int - 2, Tk)
    return p_flt

def p_upper(k_int, t_flt, tau_flt, rho_flt):
    Tk = t_k_fnc(k_int, t_flt, tau_flt, rho_flt)
    p_flt = k_int * f_k_upper(k_int - 1, Tk) - Tk * f_k_upper(k_int - 2, Tk)
    return p_flt

def d_t_k_fnc(k_int, t_flt, tau_flt, rho_flt):
    return t_flt - np.maximum(k_int,0) * tau_flt

def deriv_p_lower(k_int, t_flt, tau_flt, rho_flt):
    Tk = t_k_fnc(k_int, t_flt, tau_flt, rho_flt)
    dTk = d_t_k_fnc(k_int, t_flt, tau_flt, rho_flt)
    d_p_flt = (-k_int*f_k_lower(k_int - 1, Tk) + (k_int - 1 + Tk) * f_k_lower( k_int - 2, Tk) - Tk*f_k_lower( k_int - 3, Tk) )*dTk
    return d_p_flt

def deriv_p_upper(k_int, t_flt, tau_flt, rho_flt):
    Tk = t_k_fnc(k_int, t_flt, tau_flt, rho_flt)
    dTk = d_t_k_fnc(k_int, t_flt, tau_flt, rho_flt)
    d_p_flt = (-k_int*f_k_upper(k_int - 1, Tk) + (k_int - 1 + Tk) * f_k_upper( k_int - 2, Tk) - Tk*f_k_upper( k_int - 3, Tk) )*dTk
    return d_p_flt

def d_muller_upper(k_int, t_flt, tau_flt, rho_flt):
    k_max = (t_flt//tau_flt).astype(np.int)
    k_max = (t_flt//tau_flt).astype(np.int)
    
    # when k < k_max
    p1 = deriv_p_upper(k_int - 1, t_flt, tau_flt, rho_flt)
    p2 = deriv_p_upper(k_int, t_flt, tau_flt, rho_flt)
    p3 = deriv_p_upper(k_int + 1, t_flt, tau_flt, rho_flt)
    
    # when k == k_max
    #     p1k = p1
    #     p2k = p2
    #     p3k = -rho_flt*t_flt + (K + 1)/(1+rho_flt*tau_flt)
    ik_int = np.nonzero(k_int == k_max)
    p3[ik_int] = -t_flt[ik_int] + (k_max[ik_int] + 1)*tau_flt
    
    # when k == k_max + 1
    #     p1kp1 = 0
    #     p2kp1 = p_lower(k_int-1, t_flt, tau_flt, rho_flt)
    #     p3kp1 = rho_flt*t_flt - K/(1+rho_flt*tau_flt)
    ik_int = np.nonzero(k_int == k_max+1)
    p1[ik_int] = 0
    p2[ik_int] = -0.5*deriv_p_upper(k_int[ik_int]-1, t_flt[ik_int], tau_flt, rho_flt[ik_int])
    p3[ik_int] = t_flt[ik_int] - k_max[ik_int]*tau_flt

    p_mul_up_flt = p1 - 2 * p2 + p3
    return p_mul_up_flt
    
#     p1 = deriv_p_upper(k_int - 1, t_flt, tau_flt, rho_flt)
#     p2 = deriv_p_upper(k_int, t_flt, tau_flt, rho_flt)
#     p3 = deriv_p_upper(k_int + 1, t_flt, tau_flt, rho_flt)
#     p_mul_up_flt = p1 - 2 * p2 + p3
    return p_mul_up_flt

def d_muller_lower(k_int, t_flt, tau_flt, rho_flt):
    
    k_max = (t_flt//tau_flt).astype(np.int)
    
    # when k < k_max
    p1 = deriv_p_lower(k_int - 1, t_flt, tau_flt, rho_flt)
    p2 = deriv_p_lower(k_int, t_flt, tau_flt, rho_flt)
    p3 = deriv_p_lower(k_int + 1, t_flt, tau_flt, rho_flt)
    
    # when k == k_max
    #     p1k = p1
    #     p2k = p2
    #     p3k = -rho_flt*t_flt + (K + 1)/(1+rho_flt*tau_flt)
    ik_int = np.nonzero(k_int == k_max)
    p3[ik_int] = -t_flt[ik_int] + (k_max[ik_int] + 1)*tau_flt
    
    # when k == k_max + 1
    #     p1kp1 = 0
    #     p2kp1 = p_lower(k_int-1, t_flt, tau_flt, rho_flt)
    #     p3kp1 = rho_flt*t_flt - K/(1+rho_flt*tau_flt)
    ik_int = np.nonzero(k_int == k_max+1)
    p1[ik_int] = 0
    p2[ik_int] = -0.5*deriv_p_lower(k_int[ik_int]-1, t_flt[ik_int], tau_flt, rho_flt[ik_int])
    p3[ik_int] = t_flt[ik_int] - k_max[ik_int]*tau_flt
    
#     p1 = deriv_p_lower(k_int - 1, t_flt, tau_flt, rho_flt)
#     p2 = deriv_p_lower(k_int, t_flt, tau_flt, rho_flt)
#     p3 = deriv_p_lower(k_int + 1, t_flt, tau_flt, rho_flt)
    p_mul_lo_flt = -1*(p1 - 2 * p2 + p3)
    return p_mul_lo_flt

def p_upper_2(k_int, t_flt, tau_flt, rho_flt):
    Tk = t_k_fnc(k_int, t_flt, tau_flt, rho_flt)
    p_flt = (k_int-Tk)*f_k_upper(k_int - 2, Tk) + k_int*np.exp((k_int - 1)*np.log(Tk) - Tk - gammaln(k_int))
    return p_flt

def p_lower_2(k_int, t_flt, tau_flt, rho_flt):
    Tk = t_k_fnc(k_int, t_flt, tau_flt, rho_flt)
    p_flt = (k_int-Tk)*f_k_lower(k_int - 2, Tk) - k_int*np.exp((k_int - 1)*np.log(Tk) - Tk - gammaln(k_int))
    return p_flt


#### TCSPC (time tag) ####

def tcspc_matrices(t_lam:np.ndarray,t_det:np.ndarray,tD:float,dt:float=None):
    """
    Create matrices for calculating loss of time tagged photons
    where the loss is given by np.sum(np.dot(dQ,x)-np.log(np.dot(S,x)))
        dQ is the integration matrix and S is the sample matrix
        
    inputs:
        t_lam - time grid for the retrieved photon arrival rate
        t_det - time tags of detected photons
        tD - deadtime
        dt - grid resolution of t_lam
    outputs:
        S - sample matrix for lambda at the detected photon times
        dQ - difference in integration matrix (of photon arrival rate) 
            between current and previous photon times
    """
    if dt is None:
        dt = np.mean(np.diff(t_lam))
        
    S = (t_lam[np.newaxis,:] > t_det[:,np.newaxis])*(t_lam[np.newaxis,:] <= t_det[:,np.newaxis]+dt).astype(np.float)
    S = np.delete(S,np.where(np.sum(S,axis=1)==0),axis=0)  # delete empty rows
    
    Qint = dt*(t_lam[np.newaxis,:] <= t_det[:,np.newaxis])
    QtD = np.concatenate([np.zeros((1,t_lam.size)),dt*(t_lam[np.newaxis,:] <= (t_det[:-1,np.newaxis]+tD))],axis=0)
    dQ = Qint-QtD
    dQ = np.delete(dQ,np.where(np.sum(dQ,axis=1)==0),axis=0)  # delete empty rows
    
    return S, dQ

def tcspc_hist_residual(t_lam,t_det0,tD,dt=None):
    """
    Compute the count and residual integral histograms
    treat t_lam as the start of the histogram bin
    
    The detection histogram includes fractions
    
    """
    if dt is None:
        dt = np.mean(np.diff(t_lam))
        
   # remove points that are outside the grid
    idel = np.where(t_det0 > t_lam[-1]+dt)
    # base the timing to the start of the retrieval space
    t_det = np.delete(t_det0,idel)-t_lam[0]
    
    # There may need to be some handeling to omit the deadtime integral
    # term from the last photon (sum is up to N-1)
    
    # located upper (first full) index into time tags
    idx_tn = np.minimum(np.ceil(t_det/dt),t_lam.size-1).astype(np.int)
    # locate lower (last full) index into time tags + deadtime
    idx_td = np.minimum(np.floor((t_det+tD)/dt),t_lam.size-1).astype(np.int)
                       
    # obtain the first residual
    res_t0 = np.minimum(idx_tn-t_det/dt,1)
    res_t1 = np.minimum((t_det+tD)/dt - idx_td,1)
    
    y_d = np.zeros(t_lam.size)
    y_int = np.zeros(t_lam.size)
    
    # brute force with a for loop
    for ai,idx0 in enumerate(idx_tn):
        
        if idx0 < y_int.size-1:
            y_d[idx0-1] += 1-res_t0[ai]
            y_d[idx0] += res_t0[ai]
        else:
            y_d[idx0] = 1
            
        y_int[idx0:idx_td[ai]] += -1
        if idx0 > 0:
            y_int[idx0-1] += -res_t0[ai]
        if idx_td[ai] <= y_int.size-1:
            y_int[idx_td[ai]] += -np.minimum(res_t1[ai],1)
        
        # check if this is the last time tag
        if ai < t_det.size-1:
            # check if this is not the last time
            # tag in the laser shot
            if t_det[ai+1] < t_det[ai]:
                y_int+= 1
                
        else:
            y_int+=1
            
    return y_d, y_int

def tcspc_histogram_time_tags(timetags_lst,range_bins,tD,wrap=True):
    """
    create a gridded histogram from time tags
    inputs:
        timetags_lst - list of arrays of time tags
        range_bins - desired bin edges for histogram
        wrap - wrap deadtime effect of last profile into
            the next profile
            Defaults to True
    returns:
        2D histogram of time tags (rows: laser shot, columns: range)
        2D histogram of integral time tags (rows: laser shot, columns: range)
    """

    dt = np.mean(np.diff(range_bins))
    hist_prof = np.zeros((len(timetags_lst),range_bins.size-1))
    integral_hist_prof = np.zeros((len(timetags_lst),range_bins.size-1))
    wrap_hist = None
    for prof_index in range(len(timetags_lst)):
        yint, yp, ywrap = encode_arrival_times_wrap([timetags_lst[prof_index]],range_bins,
                              tD,dt_lam=dt,h_wrap=wrap_hist)
        if wrap:
            wrap_hist = ywrap
  
        hist_prof[prof_index,:]+=yp
        integral_hist_prof[prof_index,:]+=yint
    return hist_prof,integral_hist_prof

# def tcspc_histogram_time_tags(timetags_lst,range_bins,tD,Rf=12,wrap=False):
#     """
#     create a gridded histogram from time tags
#     inputs:
#         timetags_lst - list of arrays of time tags
#         range_bins - desired bin edges for histogram
#         wrap - wrap deadtime effect of last profile into
#             the next profile
#             Defaults to false
#     returns:
#         2D histogram of time tags (rows: laser shot, columns: range)
#         2D histogram of integral time tags (rows: laser shot, columns: range)
#     """
#     # set the profile bins

#     dt = np.mean(np.diff(range_bins))
#     # integral_hist_prof,hist_prof = encode_arrival_times(timetags_lst[prof_index],range_bins[:-1],Rf,dt,tD)
#     hist_prof = np.zeros((len(timetags_lst),range_bins.size-1))
#     integral_hist_prof = np.zeros((len(timetags_lst),range_bins.size-1))
#     for prof_index in range(len(timetags_lst)):
#         # TODO photon time tag wrapping
#         # if prof_index > 0 and wrap:
#         yint,yp = encode_arrival_times([timetags_lst[prof_index]],
#                 range_bins,12,dt,tD)
#         # TODO photon time tag wrapping
#         # else:
#         #     yint,yp = encode_arrival_times([timetags_lst[prof_index]],
#         #             range_bins,12,dt,tD,t_nm1_flt=timetags_lst[prof_index-1][-1])
#         hist_prof[prof_index,:]+=yp
#         integral_hist_prof[prof_index,:]+=yint
#     return hist_prof,integral_hist_prof

def encode_arrival_times(pat_arr_lst: List[np.ndarray], 
                         tp_arr: np.ndarray,
                         R_F_int: int,
                         delta_tp_flt: float,
                         tau_d_flt: float,
                         t_nm1_flt: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Encode the arrival times of the photons.
    
    Parameters
    ----------
    pat_arr_lst: List[np.ndarray]
       Photon arrival time. Each item in the list corresponds to a laser shot. Each 
       array element corresponds to an arrival time. 
    tp_arr: np.ndarray
        The time intervals of the catersian grid.
    R_F_int: int
        The number of bits for the intergral histogram fraction.
    delta_tp_flt: float
        The catersian grid resolution
    tau_d_flt: float
        The photon detector deadtime.
    t_nm1_flt: float
        Set the time of the time tag prior to histogram inputs
    
    Returns
    -------
    H_I_arr: np.ndarray
        The photon rate intergral histogram.
    H_P_arr: np.ndarray
        The photon count histogram.
    """
    
    # Ge the number of time interval bins
    Np_int = tp_arr.size - 1
    # Precompute some quantities
    tp_R_F_int = 2**R_F_int
    
    # # TODO photon time tag wrapping
    # # if no previous time tag is provided, initialize
    # # so that there is no deadtime effect at t=0
    # if t_nm1_flt is None:
    #     t_nm1_flt = tp_arr[-1]-2*tau_d_flt

    # ------------------------------------------------------------------------------
    # Create the intergral histogram
    # log_obj.info('Creating the intergral histogram')
    H_I_arr = np.zeros(shape=(Np_int, ))
    # For each laser pulse...
    # for pat_arr in tqdm(pat_arr_lst, desc='Laser shot', leave=True):
    for pat_arr in pat_arr_lst:
        # # TODO update for wrapping around deadtime
        # # Set $t_{n-1,k}$
        # # wrap around previous time tag to this profile time
        # t_nm1_flt = t_nm1_flt-tp_arr[-1]  
        t_nm1_flt = 0 # originally 0 but that causes blanking of the first photons
        
        # Get $t_{n,k}$. Also terminate the detected photons with 
        # by inducing a photon at the time interval boundary (accumulation 
        # point).
        # print(pat_arr)
        for cnt,t_n_flt in enumerate(np.concatenate([pat_arr, [tp_arr[-1]]])):
            # The only time when inter arrival time minus the deadtime is 
            # negative is when $t_n$ is the boundary time.
            if (t_n_flt - t_nm1_flt - tau_d_flt) < 0:
                continue

            # If the scaled fraction is less than half a bit, then the 
            # integration is rounded down to zero.
            scaled_frac_int_flt = tp_R_F_int * (t_n_flt - t_nm1_flt - tau_d_flt) / delta_tp_flt
            if scaled_frac_int_flt <= 0.5:
                # Set $t_{n-1,k}$
                t_nm1_flt = t_n_flt
                continue

            # # TODO photon time tag wrapping
            # np_int = np.argmin(tp_arr < t_nm1_flt + tau_d_flt) - 1
            if cnt > 0:
                # Find $\tilde{t}_{n'}$ and $\tilde{t}_{n''}$
                np_int = np.argmin(tp_arr < t_nm1_flt + tau_d_flt) - 1
                # np_int = int((t_nm1_flt + tau_d_flt) / delta_tp_flt)
            else:
                # special case for first photon, integral starts at zero
                np_int = np.argmin(tp_arr)  

            npp_int = np.argmin(tp_arr < t_n_flt) - 1
            # npp_int = int(t_n_flt / delta_tp_flt)

            # if np.sum(tp_arr < t_n_flt) == tp_arr.size:
            #     npp_int = tp_arr.size-2
            
            # if npp_int < 0:
            #     print(tp_arr < t_n_flt)

            # print('%e,%d'%(t_nm1_flt,np_int))
            # print('%e,%d'%(t_n_flt,npp_int))

            # If $\tilde{t}_{n'} = \tilde{t}_{n''}
            if np_int == npp_int:
                a_np_flt = np.ceil(scaled_frac_int_flt) / tp_R_F_int
                H_I_arr[np_int] += a_np_flt

            # else $\tilde{t}_{n'} < \tilde{t}_{n''}$
            elif npp_int > np_int:
                if cnt > 0:
                    # this assert only applies to photons after the first one
                    assert(tp_arr[np_int + 1] >= (t_nm1_flt + tau_d_flt))
                    a_np_flt = tp_arr[np_int + 1] - t_nm1_flt - tau_d_flt
                else:
                    a_np_flt = tp_arr[np_int + 1] - t_nm1_flt
                    
                assert(t_n_flt >= tp_arr[npp_int])
                a_npp_flt = t_n_flt - tp_arr[npp_int]

                H_I_arr[np_int] += np.ceil(tp_R_F_int * a_np_flt / delta_tp_flt) / tp_R_F_int
                for tld_n_int in range(np_int + 1, npp_int):
                    H_I_arr[tld_n_int] += 1
                H_I_arr[npp_int] += np.ceil(tp_R_F_int * a_npp_flt / delta_tp_flt) / tp_R_F_int
                
            else:
                # when the program ends up here, it is likely
                # because a time tag exceeded the maximum 
                # histogram value.  It probably does not mean we
                # should raise an error, but just drop the count
                # without recording it
                # t_n_flt should proably not be updated either
                print('t_nm1_flt: %e'%t_nm1_flt)
                print('t_n_flt: %e'%t_n_flt)
                print('np_int: %d'%np_int)
                print('npp_int: %d'%npp_int)
                print('tp_arr[-1]: %e'%tp_arr[-1])
                # raise RuntimeError('This should not have happened.')
                continue

            # Set $t_{n-1,k}$
            t_nm1_flt = t_n_flt
            
    # ------------------------------------------------------------------------------
    # Create the photon counting histogram
    # log_obj.info('Creating the photon counting histogram')
    H_P_arr = np.zeros(shape=(Np_int, ))
    # For each laser pulse...
    # for pat_arr in tqdm(pat_arr_lst, desc='Laser shot', leave=True):
    for pat_arr in pat_arr_lst:
        for t_n_flt in pat_arr:
            idx = np.minimum(int(t_n_flt / delta_tp_flt),H_P_arr.size-1)
            H_P_arr[idx] += 1
    
    return H_I_arr, H_P_arr

def encode_arrival_times_wrap(t_det_lst:list,t_lam_edges:np.ndarray,tD:float,
                              dt_lam:float=None,h_wrap:np.ndarray=None)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode photon time tags into histogram data
    
    inputs:
        t_det_lst: list
            list of arrays where each array contains all the time tags from a laser shot.
        t_lam_edges: np.ndarray
            1D array of the bin edges to be used in the histogram
        tD: float
            detector dead time
        dt_lam: float
            time resolution of the histogram
        h_wrap: np.ndarray
            deadtime histogram from previous shot to be folded into this histogram.
    
    returns
        h_int: np.ndarray
            a 1D histogram of the integral histogram used in processing
        h_d: np.ndarray
            a 1D histogram of photon arrival times.  The standard lidar histogram.
        h_wrap: np.ndarray
            an integral histogram of deadtime effects that wrap around to the next profile.
            Not used in actual data processing, but needed for successive profiles.
            
    
    """
    
    if dt_lam is None:
        dt_lam = np.mean(np.diff(t_lam_edges))

    if h_wrap is not None:
        # update the wrap around deadtime if provided
        h_wrap_size = h_wrap.size
    else:
        h_wrap_size = np.int(np.ceil(tD/dt_lam))
        h_wrap = np.zeros(h_wrap_size)
    
    h_d = np.zeros(t_lam_edges.size-1)
    h_det = np.zeros(t_lam_edges.size-1+h_wrap_size)  # initialize integral histogram with extra bins
    h_det[:h_wrap_size] = h_wrap
                             
    for shot_idx in range(len(t_det_lst)):
        

        # TODO: update h_det based on h_wrap from the previous run

        nmax = t_lam_edges.size-1 # index to split histogram
        # h_wrap = np.zeros(h_det.size-nmax)  # histogram that wraps into the next histogram
        
        # broadcast most of the operations
        n_det_arr = np.ceil((t_det_lst[shot_idx]-t_lam_edges[0])/dt_lam).astype(np.int)  # interger position of first full observation
        remain_det_arr = 1-np.remainder((t_det_lst[shot_idx]-t_lam_edges[0]),dt_lam)/dt_lam  # area under curve of first bin
        n_deadtime_arr = np.floor((t_det_lst[shot_idx]+tD-t_lam_edges[0])/dt_lam).astype(np.int)  # last full integer position of deadtime integral
        remain_deadtime_arr = np.remainder(t_det_lst[shot_idx]+tD-t_lam_edges[0],dt_lam)/dt_lam  # remainder in last deadtime integral
        n_delta_arr = (n_deadtime_arr-n_det_arr).astype(np.int)

        for ph_idx,n_det in enumerate(n_det_arr):
            if n_delta_arr[ph_idx] == -1:
                # special case where t and t+tD are in the same bin
                h_det[n_det-1] -= (remain_deadtime_arr[ph_idx]-(1-remain_det_arr[ph_idx]))

            else:
                h_det = np.roll(h_det,-(n_det-1))
                h_det[0] -= remain_det_arr[ph_idx]
                h_det[1:(n_delta_arr[ph_idx]+1)] -= 1
                h_det[n_delta_arr[ph_idx]+1] -= remain_deadtime_arr[ph_idx]
                h_det = np.roll(h_det,n_det-1,axis=0)
                
        if shot_idx != len(t_det_lst):
            # wrap deadtime around unless this is the last shot in the list
            h_det[:h_wrap.size]+=h_det[nmax:]
            h_det[nmax:] = 0
        
        # standard photon count histogram
        h_d += np.histogram(t_det_lst[shot_idx],bins=t_lam_edges)[0]

    h_int = h_det[:nmax]   # store the integral histogram
    h_wrap = h_det[nmax:]  # capture the wrap around from the last shot
    h_int = h_int+1        # update to account for how we capture the integrals
    
    return h_int, h_d, h_wrap

def regrid_histogram(y:np.ndarray,t_mult:int,r_mult:int)->np.ndarray:
    """
    Regrid a histogram to a larger grid that is integer multiples
    of the orginal resolution
    """
    yp_tmp = y.copy()

    # regrid time
    yp_cum_tmp = np.cumsum(yp_tmp,axis=0)[(t_mult-1)::t_mult,:]
    yp_tmp = np.concatenate([yp_cum_tmp[:1,:],np.diff(yp_cum_tmp,axis=0)],axis=0)
    
    # regrid range
    yp_cum_tmp = np.cumsum(yp_tmp,axis=1)[:,(r_mult-1)::r_mult]
    yp_tmp = np.concatenate([yp_cum_tmp[:,:1],np.diff(yp_cum_tmp,axis=1)],axis=1)

    return yp_tmp

def timebin_time_tags(tt:np.ndarray,r_bins:np.ndarray,tD:float,sync_count:int,dr:float)->Tuple[np.ndarray,np.ndarray]:
    """
    tt - time tags to be histogrammed
    r_bins - bins for the histogram
    tD - detector dead time
    sync_count - number of sync pulses included in tt data
    dr - range bin resolution

    returns:
        h_d - standard detected counts histogram
        h_int - active shots (integral) histogram
    """

    idel = np.where(tt >= r_bins[-1])
    tt = np.delete(tt,idel)

    ttd = tt + tD
    imax = np.where(ttd > r_bins[-1])
    deadtime_wrap = ttd[imax] - r_bins[-1]
    ttd[imax] = r_bins[-1]


    h_int_dead = np.zeros(r_bins.size-1)
    h_int_dead += np.sum((r_bins[:-1,None]-dr >= tt[None,:]) & (r_bins[1:,None] < ttd[None,:]),axis=1)

    tt_idx = np.searchsorted(r_bins,tt)
    residual_tt = (r_bins[tt_idx+1]-tt)/dr
    h_int_dead[tt_idx]+= residual_tt

    ttd_idx = np.searchsorted(r_bins,ttd)
    residual_ttd = 1-(r_bins[ttd_idx]-ttd)/dr
    h_int_dead[ttd_idx-1]+= residual_ttd
    
    h_d = np.histogram(tt,bins=r_bins)[0]
    h_int = sync_count - h_int_dead
    
    return h_d, h_int

def dt64_to_datetime(t:np.datetime64)->datetime.datetime:
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    start_seconds_since_epoch = (t - unix_epoch) / one_second
    t_dt = datetime.datetime.utcfromtimestamp(start_seconds_since_epoch)
    return t_dt


def time_down_sample(a:np.ndarray,down_int:int):
    """
    down sample an array in time (0 axis) by a factor of down_int
    the down sampling is performed by summing (not averaging)
    a: np.ndarray
        array to be down sampled
    down_int: int
        down sampling factor
    """
    if down_int > 1:
        cat_len = int(np.ceil(a.shape[0]/down_int)*down_int-a.shape[0])
        if cat_len > 0:
            cat_nan = np.ones((cat_len,a.shape[1]))*np.nan
            a_full = np.concatenate((a,cat_nan),axis=0)
        else:
            a_full = a
            
        a_down = np.nansum(a_full.T.reshape(a_full.shape[1],a_full.shape[0]//down_int,-1),axis=2).T
    else:
        a_down = a
        
    return a_down

def time_down_sample_mean(a:np.ndarray,down_int:int):
    """
    down sample an array in time (0 axis) by a factor of down_int
    the down sampling is performed by averaging
    a: np.ndarray
        array to be down sampled
    down_int: int
        down sampling factor
    """
    if down_int > 1:
        cat_len = int(np.ceil(a.shape[0]/down_int)*down_int-a.shape[0])
        if cat_len > 0:
            cat_nan = np.ones((cat_len,a.shape[1]))*np.nan
            a_full = np.concatenate((a,cat_nan),axis=0)
        else:
            a_full = a
            
        a_down = np.nanmean(a_full.T.reshape(a_full.shape[1],a_full.shape[0]//down_int,-1),axis=2).T
    else:
        a_down = a
        
    return a_down

def range_down_sample(a:np.ndarray,down_int:int):
    """
    down sample an array in range (1 axis) by a factor of down_int
    the down sampling is performed by summing (not averaging)
    a: np.ndarray
        array to be down sampled
    down_int: int
        down sampling factor
    """
    if down_int > 1:
        cat_len = int(np.ceil(a.shape[1]/down_int)*down_int-a.shape[1])
        if cat_len > 0:
            cat_nan = np.ones((a.shape[0],cat_len))*np.nan
            a_full = np.concatenate((a,cat_nan),axis=1)
        else:
            a_full = a
            
        a_down = np.nansum(a_full.reshape(a_full.shape[0],a_full.shape[1]//down_int,-1),axis=2)
    else:
        a_down = a
    
    return a_down

def range_down_sample_mean(a:np.ndarray,down_int:int):
    """
    down sample an array in range (1 axis) by a factor of down_int
    the down sampling is performed by averaging
    a: np.ndarray
        array to be down sampled
    down_int: int
        down sampling factor
    """
    if down_int > 1:
        cat_len = int(np.ceil(a.shape[1]/down_int)*down_int-a.shape[1])
        if cat_len > 0:
            cat_nan = np.ones((a.shape[0],cat_len))*np.nan
            a_full = np.concatenate((a,cat_nan),axis=1)
        else:
            a_full = a
            
        a_down = np.nanmean(a_full.reshape(a_full.shape[0],a_full.shape[1]//down_int,-1),axis=2)
    else:
        a_down = a
    
    return a_down

def time_up_sample(a:np.ndarray,up_int:int,t_size_int:int=None):
    """
    up sample an array in time (0 axis) by a factor of up_int
    the up sampling is performed by repeating the values
    
    a: np.ndarray
        array to be up sampled
    up_int: int
        up sampling factor
    t_size_int: int
        size of the output array along the time axis
    """
    a_up = np.repeat(a,up_int,axis=0)
    if t_size_int is None:
        return a_up
    else:
        a_up = a_up[:t_size_int,:]
    return a_up

def range_up_sample(a:np.ndarray,up_int:int,r_size_int:int):
    """
    up sample an array in range (1 axis) by a factor of up_int
    the up sampling is performed by repeating the values
    
    a: np.ndarray
        array to be up sampled
    up_int: int
        up sampling factor
    r_size_int: int
        size of the output array along the range axis
    """
    a_up = np.repeat(a,up_int,axis=1)
    if r_size_int is None:
        return a_up
    else:
        a_up = a_up[:,:r_size_int]
    return a_up
