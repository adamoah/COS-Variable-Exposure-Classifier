# custom cos functions for parsing fits and log files

# import necessary modules (will add more as needed)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# astropy submodules
import astropy
from astropy.table import Table
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

# scipy submodules
import scipy
from scipy.ndimage import convolve1d
from scipy.signal.windows import gaussian

# regex module
import re

# warings module
import warnings

# os
import os


def extract_from_file(path, ext, header=False, method='TABLE'):

    '''
    Returns either a header or data object from a valid fits data filepath
    Parameters
    ----------
    path: str, path for the fits file.
    ext: int, retrieve from a specified HDU extention.
    header: bool, whether to extract the header of the fits file or the data. Default is False.
    method: str, method for extracting fits data either as a fits record (FITS) or astropy table (TABLE). Default is TABLE.
    '''
    if header:
        return fits.getheader(path, ext=ext)

    if method == 'FITS':
        try: # ty to extract the data from the given ext if no data exists get retrieve from the default ext
            return fits.getdata(path, ext=ext)
        except IndexError:
            print(f"ERROR! No data present in HDU extention #{ext} getting data from alterative extention")
            return fits.getdata(path)
            
    elif method == 'TABLE':
        try: # ty to extract the data from the given ext if no data exists get retrieve from the default ext
            return Table.read(path, hdu=ext)
        except ValueError:
            print(f"ERROR! No table present in HDU extention #{ext} getting data from alternatve extention")
            with warnings.catch_warnings(): # suppress astropy warning to reduce output clutter
                warnings.simplefilter('ignore', AstropyWarning)
                return Table.read(path)


def plot_x_vs_y(xcorr, ycorr, name, boxes=None, box_names=None, display=False, save=None):
    '''
    generates a scatter plot of the xcorr and ycorr pixel positions, and saves the plot to the output directory
    Parameters
    ----------
    xcorr: ndarray, corrected x pixel positions.
    ycorr: ndarray, corrected y pxiel positions.
    name: str, name of the fits file the data originated from.
    boxes: ndarray, list of bounding box locations. Default is None.
    box_names: ndarray, list of bounding box names. Defaults is None
    display: bool, weather to display the plot. Defaults is False.
    save: bool, whether or not to save the plot to the outputs directory. Default is False
    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    ax.scatter(xcorr, ycorr, s=1/50)
    ax.set_title("Corrected X vs. Y Photon Counts " + name, size=20)
    ax.set_xlabel("X Pixel Coord", size=20)
    ax.set_ylabel("Y Pixel Coord", size=20)

    if boxes is not None:
        for i, (box, bname) in enumerate(zip(boxes, box_names)): # highlight XTRACTAB bounding boxes
            ax.axhspan(box[0], box[1], color='cmykrb'[i], alpha=0.3 , label=bname)
    
    ax.legend(loc='upper right')
    fig.tight_layout()

    if not display:
        plt.close(fig)
    else:
        plt.show()
    if save is not None:
        fig.savefig(save + name + "_xcorr_vs_ycorr.png")


def plot_distribution_by_location(xcorr, ycorr, name, display=False, save=None):
    '''
    plots a histogram of the number of photon events that occur at each xcorr and ycorr pixel location ands saves
    the plot if specified
    Parameters
    ----------
    xcorr: ndarray, x axis pixel location
    ycorr: ndarray, y axis pixel location
    name: str, name of the fits file where the data originated from
    display: bool, weather to display the plot. Defaults is False.
    save: bool, whether to save the plot or not. Default is False.
    '''
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 11))
    
    ax0.hist(xcorr, bins=len(np.unique(xcorr)), color='b', label='XCORR')
    ax0.set_xlabel('Dispersion Axis', size=15)
    ax0.set_ylabel('Photon Event Counts', size=15)
    ax0.legend(loc='upper right')
    ax0.set_title('Distribution of Photon events by pixel location ' + name, size=15)
    
    ax1.hist(ycorr, bins=len(np.unique(ycorr)), color='r', label='YCORR')
    ax1.set_xlabel('Cross Dispersion Axis', size=15)
    ax1.set_ylabel('Photon Event Counts', size=15)
    ax1.legend(loc='upper right')

    fig.tight_layout()

    if not display:
        plt.close(fig)
    else:
        plt.show()
    if save is not None:
        fig.savefig(save + name + '_distribution.png')

def plot_cross_dispersion(ycorr, name, boxes=None, box_names=None, display=False, save=False):
    '''
    computes and plots counts on the cross dispersion axis with XCTRACTAB bounding boxes and saves the plot if specified
    Parameters
    ----------
    ycorr: ndarray, y axis pixel location
    name: str, name of the fits file where the data originated from
    boxes: ndarray, list of bounding box locations
    box_names: ndarray, list of bounding box names
    display: bool, weather to display the plot. Defaults is False.
    save: bool, whether to save the plot ot not. Default is False.
    '''
    # get unique corrdinates and empty counts
    unique = np.unique(ycorr)
    counts = np.zeros(shape=(len(unique),))
    for i, loc in enumerate(unique):
        counts[i] = np.count_nonzero(ycorr == loc)
    
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    
    ax.plot(counts, unique)
    if boxes is not None:
        for i, (box, bname) in enumerate(zip(boxes, box_names)):
            ax.axhspan(box[0], box[1], color='cmykrb'[i], alpha=0.3 , label=bname)
    
    ax.set_ylabel('Cross Dispersion Axis', size=15)
    ax.set_xlabel('Counts', size=15)
    ax.set_title('Number of Photons on Cross Dispersion Axis ' + name, size=20)
    ax.legend(loc='lower right')
    
    fig.tight_layout()

    if not display:
        plt.close(fig)
    else:
        plt.show()
    if save is not None:
        fig.savefig(save + name + '_yaxis.png')

def get_count_rate_vs_time(time, ycorr, wl, bounds, exptime, bin_size=0.64, verbose=0):
    '''
    Computes and returns the count rate of photon events at each timestep according to some bins.
    Returns both variable bins (based on global count rate) and fixed bins data
    Parameters
    ----------
    time: ndarray, array representing the timesteps for photon events,
    ycorr: ndarray, pixel locations of photons on cross dispersion axis
    wl: ndarray, wavelength of the individual photons recorded by the detector
    exptime: float, how the long the exposure was (max of EXPTIME and PLANTIME)
    bounds: tuple, extraction box bounds for the PSA region.
    bin_size: float, the time in seconds to bin the counts. Default is 0.64
    verbose: bool, print out more information about what the function is doing. Default is False.
    '''

    # initialize the final arrays
    unique_timesteps = np.arange(0, exptime, bin_size)
    unique_counts = np.ndarray(shape=(len(unique_timesteps),))
    if verbose >= 1:
        print(f"Length of Timesteps: {len(unique_timesteps)}")
        print(f"Start: {min(time)} s | Stop: {exptime} s | Step: {bin_size} s")
        print("Beginning computing counts")

    wl_indicies = []
    for x_wl in [1216, 1302, 1200, 1026, 989, 1356]:
        wl_indicies.append((wl < x_wl - 7) | (wl > x_wl + 7))
    
    y_indicies = [] # each row represents boolean array for one of the PSA regions
    for bound in bounds: # filter for photons within in the PSA regions
        y_indicies.append((ycorr >= bound[0]) & (ycorr <= bound[1]))  

    # boolean and each row in the w and y_indicies arrays
    y_indicies = np.logical_or.reduce(y_indicies, axis=0)
    wl_indicies = np.logical_and.reduce(wl_indicies, axis=0)
    
    # now contains which photon counts belong to PSA and those not in airglow wavelengths
    time = time[(y_indicies & wl_indicies)]
    
    # return the idx of unique_timesteps each time t belongs to such that unique_timesteps(i-1) <= t < unique_timesteps(i) 
    bin_indicies = np.digitize(time, bins=unique_timesteps - 0.0001, right=False)
    unique_counts = np.bincount(bin_indicies)[1:] # count occurences of each bin (ignore first value it is always zero)
    
    return unique_counts, unique_timesteps

def plot_count_rate(timesteps, counts, name, display=False, save=None):
    '''
    plots a scatter plot of the count rates at each timestep, and saves the plot if specified
    Parameters
    ----------
    timesteps: ndarray, timesteps where photon events were recorded
    counts: ndarray, the number of photon events recorded at each timestep
    name: str, name of the fits file the data originated from
    display: bool, weather to display the plot. Defaults is False.
    save: bool, whether to save the plot. Default is False.
    '''
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    mean = np.mean(counts)
    std = np.std(counts)
    ax.plot(timesteps, counts, color='k', marker='.', label=f'Count Rate (bins:{len(timesteps)})')
    ax.axhline(mean, color='r', linewidth=3, label='Average')
    ax.axhline(mean+std, color='orange', linewidth=3, label='Standard Deviation')
    ax.axhline(mean-std, color='orange', linewidth=3, label='Standard Deviation')
    ax.set_xlabel('Time (Seconds)', size=20)
    ax.set_ylabel('Number of Photon Events', size=20)
    title = f"Count Rate of Photon Events {name}"
    ax.set_title(title, size=20)
    ax.legend(loc='upper left')
    
    fig.tight_layout()

    if not display:
        plt.close(fig)
    else:
        plt.show()
    if save is not None:
        fig.savefig(save + name + "_raw.png")


def plot_dual_count_rate(timesteps, counts, f_timesteps, f_counts, name, display=False, save=None):
    fig, (ax0, ax1) = plt.subplots (2, 1, figsize=(13, 12)) # create figure
    # statistical info
    mean, std = np.mean(counts), np.std(counts)
    f_mean, f_std = np.mean(f_counts), np.std(f_counts)

    # variable bin plots
    ax0.plot(timesteps, counts, color='k', marker='.', label=f'Count Rate (bins:{len(timesteps)})')
    ax0.axhline(mean, color='r', linewidth=3, label='Average')
    ax0.axhline(mean+std, color='orange', linewidth=3, label='Standard Deviation')
    ax0.axhline(mean-std, color='orange', linewidth=3)
    ax0.legend(loc='upper right')

    # fixed bin plots
    ax1.plot(f_timesteps, f_counts, color='k', marker='.', label=f'Count Rate (bins:{len(f_timesteps)})')
    ax1.axhline(f_mean, color='r', linewidth=3, label='Average')
    ax1.axhline(f_mean+f_std, color='orange', linewidth=3, label='Standard Deviation')
    ax1.axhline(f_mean-f_std, color='orange', linewidth=3)
    ax1.legend(loc='upper right')

    fig.suptitle(f"Count Rate of Photon Events {name}", size=20)
    fig.supylabel('Number of Photon Events', size=20)
    fig.supxlabel('Time (Seconds)', size=20)

    fig.tight_layout()

    if not display:
        plt.close(fig)
    else: # display plot
        plt.show()
    if save is not None: # save
        fig.savefig(save + name + "_dual_count_rate.png")
        
def find_all_removed(log_filepath, verbose=0):
    '''
    Finds and returns a list of removed files from a output log file
    based on a regex match
    Parameters
    ----------
    log_filepath: str, filepath for the output logfile to parse
    verbose: bool, print out information about what the function is doing. Default is False.
    '''
    files = []
    # context manager for automatic file handing
    with open(log_filepath, "r", encoding="utf-8") as out:
        data = out.read() # read file data
        if verbose >= 1:
            print("Matching regex: 'Removing file (.+/)*(\\w+\\.fits) .*'\n--------------------------------------") # pattern to match
        for i, line in enumerate(data.splitlines()): # iterate each line in the file
            match = re.search("Removing file (.+/)*(\\w+\\.fits) .*", line) # regex match
            
            if match != None and verbose == 2: # print info about matched lines
                print(f"matched string '{match.group()}' from line {i+1}. Adding file '{match.group(2)}' to output.")
                files.append(match.group(2))
            elif match!= None: # add removed file to final output
                files.append(match.group(2))

    if verbose >= 1: # print all collected files
        print(f"Removed Files compiled:\n{files}")
    return files

# taken from the Extract training notebook
def readxtractab(xtractab, grat, cw, aper, segment):

    """
    Reads in an XTRACTAB row of a particular COS mode and\
    returns extraction box sizes and locations.
    Inputs:
    xtractab (str) : path to xtractab file.
    raw (bool) : default False, meaning that the data is assumed to be corrtag.
    grat (string) : grating of relavent row (i.e. "G185M")
    cw (int or numerical) : cenwave of relavent row (i.e. (1786))
    aper (str) : aperture of relavent row (i.e. "PSA")
    segment (str) : detector segment from which the exposure came from ('N/A', 'FUVA', or 'FUVB')
    Returns:
    y locations of bottoms/tops of extraction boxes
        if NUV: stripe NUVA/B/C, and 2 background boxes
        elif FUV: FUVA/B, and 2 background boxes for each FUVA/B.
    """
    with fits.open(xtractab) as f:
        xtrdata = f[1].data # Read the fits data
    
    
    if segment == 'NUV': # Then NUV data:
        sel_nuva = np.where((xtrdata['segment'] == 'NUVA') & # Find NUVA 
                            (xtrdata['aperture'] == aper) & # of the right row
                            (xtrdata['opt_elem'] == grat) &
                            (xtrdata['cenwave'] == cw))

        sel_nuvb = np.where((xtrdata['segment'] == 'NUVB') & # Now NUVB
                            (xtrdata['aperture'] == aper) &
                            (xtrdata['opt_elem'] == grat) &
                            (xtrdata['cenwave'] == cw))

        sel_nuvc = np.where((xtrdata['segment'] == 'NUVC') & # Now NUVC
                            (xtrdata['aperture'] == aper) &
                            (xtrdata['opt_elem'] == grat) &
                            (xtrdata['cenwave'] == cw))

        hgta = xtrdata['HEIGHT'][sel_nuva][0] # Find heights
        hgtb = xtrdata['HEIGHT'][sel_nuvb][0] #  of spec extract boxes
        hgtc = xtrdata['HEIGHT'][sel_nuvc][0]

        bspeca = xtrdata['B_SPEC'][sel_nuva][0] # y-intercept (b) of spec 
        bspecb = xtrdata['B_SPEC'][sel_nuvb][0] #  boxes
        bspecc = xtrdata['B_SPEC'][sel_nuvc][0]

        boundsa = [bspeca - hgta/2, bspeca + hgta/2] # Determine y bounds of boxes 
        boundsb = [bspecb - hgtb/2, bspecb + hgtb/2]
        boundsc = [bspecc - hgtc/2, bspecc + hgtc/2]

        bkg1a = xtrdata['B_BKG1'][sel_nuva][0] # Do the same for the bkg extract boxes
        bkg2a = xtrdata['B_BKG2'][sel_nuva][0]
        bhgta = xtrdata['BHEIGHT'][sel_nuva][0]
        bkg1boundsa = [bkg1a - bhgta/2, bkg1a + bhgta/2]
        bkg2boundsa = [bkg2a - bhgta/2, bkg2a + bhgta/2]

        # The background locations are by default the same for all stripes

        return (boundsa, boundsb, boundsc, bkg1boundsa, bkg2boundsa),('NUVA','NUVB','NUVC','BKG-1','BKG-2')
    
    elif segment == 'FUVA': # Then FUVA data:
        sel_fuva = np.where((xtrdata['segment'] == 'FUVA') & # Find FUVA 
                            (xtrdata['aperture'] == aper) &  # of the right row
                            (xtrdata['opt_elem'] == grat) &
                            (xtrdata['cenwave'] == cw))

        hgta = xtrdata['HEIGHT'][sel_fuva][0] # Find heights
        
        bspeca = xtrdata['B_SPEC'][sel_fuva][0] # y-intercept (b) of spec 
        
        boundsa = [bspeca - hgta/2, bspeca + hgta/2] # determine y bounds of boxes 

        bkg1a = xtrdata['B_BKG1'][sel_fuva][0] # Do the same for the bkg extract boxes
        bkg2a = xtrdata['B_BKG2'][sel_fuva][0]
        

        try:
            bhgta = xtrdata['BHEIGHT'][sel_fuva][0]
            bkg1boundsa = [bkg1a - bhgta/2, bkg1a + bhgta/2]
            bkg2boundsa = [bkg2a - bhgta/2, bkg2a + bhgta/2]
            
        except KeyError:
            bhgt1a = xtrdata['B_HGT1'][sel_fuva][0]
            bhgt2a = xtrdata['B_HGT2'][sel_fuva][0]
            bkg1boundsa = [bkg1a - bhgt1a/2, bkg1a + bhgt1a/2]
            bkg2boundsa = [bkg2a - bhgt2a/2, bkg2a + bhgt2a/2]
            
    
        return (boundsa, bkg1boundsa, bkg2boundsa), ('FUVA','BKG-1A','BKG-2A')

    elif segment == 'FUVB': # Then FUVB data
        sel_fuvb = np.where((xtrdata['segment'] == 'FUVB') & # Now FUVB
                            (xtrdata['aperture'] == aper) &
                            (xtrdata['opt_elem'] == grat) &
                            (xtrdata['cenwave'] == cw))

        hgtb = xtrdata['HEIGHT'][sel_fuvb][0] # Find Heights

        bspecb = xtrdata['B_SPEC'][sel_fuvb][0] # y intercept

        boundsb = [bspecb - hgtb/2, bspecb + hgtb/2] # y bounds

        bkg1b = xtrdata['B_BKG1'][sel_fuvb][0] # Do the same for the bkg extract boxes
        bkg2b = xtrdata['B_BKG2'][sel_fuvb][0]

        try:
            bhgtb = xtrdata['BHEIGHT'][sel_fuvb][0]
            bkg1boundsb = [bkg1b - bhgtb/2, bkg1b + bhgtb/2]
            bkg2boundsb = [bkg2b - bhgtb/2, bkg2b + bhgtb/2]
            
        except KeyError:
            bhgt1b = xtrdata['B_HGT1'][sel_fuvb][0]
            bhgt2b = xtrdata['B_HGT2'][sel_fuvb][0]
            bkg1boundsb = [bkg1b - bhgt1b/2, bkg1b + bhgt1b/2]
            bkg2boundsb = [bkg2b - bhgt2b/2, bkg2b + bhgt2b/2]

        return (boundsb, bkg1boundsb, bkg2boundsb), ('FUVB','BKG-1B','BKG-2B')

    elif segment == 'BOTH': # Both FUVA and FUVB data
        sel_fuva = np.where((xtrdata['segment'] == 'FUVA') & # Find FUVA 
                            (xtrdata['aperture'] == aper) &  # of the right row
                            (xtrdata['opt_elem'] == grat) &
                            (xtrdata['cenwave'] == cw))

        sel_fuvb = np.where((xtrdata['segment'] == 'FUVB') & # Now FUVB
                            (xtrdata['aperture'] == aper) &
                            (xtrdata['opt_elem'] == grat) &
                            (xtrdata['cenwave'] == cw))
        
        hgta = xtrdata['HEIGHT'][sel_fuva][0] # Find heights
        hgtb = xtrdata['HEIGHT'][sel_fuvb][0] #   of spec extract boxes
        bspeca = xtrdata['B_SPEC'][sel_fuva][0] # y-intercept (b) of spec 
        bspecb = xtrdata['B_SPEC'][sel_fuvb][0] #  boxes
        boundsa = [bspeca - hgta/2, bspeca + hgta/2] # determine y bounds of boxes 
        boundsb = [bspecb - hgtb/2, bspecb + hgtb/2]

        bkg1a = xtrdata['B_BKG1'][sel_fuva][0] # Do the same for the bkg extract boxes
        bkg2a = xtrdata['B_BKG2'][sel_fuva][0]
        bhgt1a = xtrdata['B_HGT1'][sel_fuva][0]
        bhgt2a = xtrdata['B_HGT2'][sel_fuva][0]
        bkg1boundsa = [bkg1a - bhgt1a/2, bkg1a + bhgt1a/2]
        bkg2boundsa = [bkg2a - bhgt2a/2, bkg2a + bhgt2a/2]
        
        bkg1b = xtrdata['B_BKG1'][sel_fuvb][0] # Do the same for the bkg extract boxes
        bkg2b = xtrdata['B_BKG2'][sel_fuvb][0]
        bhgt1b = xtrdata['B_HGT1'][sel_fuvb][0]
        bhgt2b = xtrdata['B_HGT2'][sel_fuvb][0]
        bkg1boundsb = [bkg1b - bhgt1b/2, bkg1b + bhgt1b/2]
        bkg2boundsb = [bkg2b - bhgt2b/2, bkg2b + bhgt2b/2]

        return (boundsa, boundsb, bkg1boundsa, bkg2boundsa, bkg1boundsb, bkg2boundsb), ('FUVA','FUVB','BKG-1A','BKG-2A','BKG-1B','BKG-2B')

def get_outlier_segments(idx_arr):
    '''
    caluclates and returns the length of the outlier sections and their positions
    Parameters
    ----------
    outliers: ndarray, array indicate which timesteps/count rates are outliers (-1) and inliners (1).
    '''

    idx_positions = [] # stores start and end index for each segment
    segment_lengths = [] 
    if len(idx_arr) > 0: # only iterate if there are outliers
        start = idx_arr[0]
        end = start
        for j in range(1, len(idx_arr)): # iterate through and count consecutive indicies
            # if the current index is the next in the sequence continue iterating
            if idx_arr[j] == end + 1:
                end = idx_arr[j]
            else: # if current index is not consecutive then it is a new segment
                idx_positions.append((start, end))
                segment_lengths.append(end - start + 1)
                start = idx_arr[j]
                end = start
                
            if j == len(idx_arr) - 1: # last index either terminates count or is 1
                idx_positions.append((start, end))
                segment_lengths.append(end - start + 1)

    # convert to numpy array, use default values if no data was found
    segment_lengths = np.array(segment_lengths) if len(segment_lengths) > 0 else np.array([1])
    idx_positions = np.array(idx_positions) if len(idx_positions) > 0 else np.array([[0, 0]])

    return segment_lengths, idx_positions

def get_derivative_positions(counts, ws=20, std=5, thresh=0.75, min_dist=8, pad_p=0.05):
    '''
    Computes and returns the start and end positions of a rise/drop using the derivative method.
    Parameters
    ----------
    counts: ndarray, binned photon count_rates
    z_mean: float, z_score mean of the outliers
    ws: int, size of the gaussan filter. Default is 20
    std: int, standard deviation of the gaussian filter. Default is 5
    thresh: float, ratio threshold of the magintude of the start and end derivative. If the ratio is less than the threshold, the interval is bounded by the start or end of the exposure. Default is 0.75
    min_dist: int, minimum distance, in indicies, between the start and end derivative. If the distance is less than the minimum, the interval is bounded by the start or end of the exposure. Default is 8
    pad: float, how many indicies to pad the start and end index as a percentage of the computed interval. Default is 0.05.
    '''
    
    # smooth count rate signal and get gradient magnitudes
    sliding_window = gaussian(ws, std=std)
    smooth_x = convolve1d(counts, sliding_window, mode='reflect') / sum(sliding_window) 
    deriv = np.abs(np.gradient(smooth_x))

    max_idxs = np.argpartition(deriv, len(deriv)-2)[-2:] # find the positions of the two highest magnitudes in the derivative
    min_, max_ = np.min(max_idxs), np.max(max_idxs)
    
    # if the deriv magnitudes are not close enough or the distance is too short
    # set one of the bounds to the start or end of the array (whichever is closer)
    if (((deriv[max_idxs[0]] / deriv[max_idxs[1]]) < thresh) and ((max_ - min_) < min_dist)):
        min_ = min_ if min_ >= (len(deriv) - 1 - max_) else 0
        max_ = max_ if min_ < (len(deriv) - 1 - max_) else (len(deriv) - 1)

    pad = int(pad_p * (max_ - min_))
    # pad start and end use clip to ensure they do not go outside valid range
    idxs = np.clip([min_ - pad, max_ + pad], a_min=0, a_max=(len(deriv) - 1))

    return idxs[0], idxs[1]

def get_max_zscore(z_score, segment_lengths):
    '''
    computes and returns the z-score of the maximum segment
    Parameters
    ----------
    z_score: array, the z_scores for all outliers detected by LOF
    segment_lengths: array, the lengths of each continuous outlier segment
    '''

    zscore_mean = None # will store the mean of the max segment

    if len(segment_lengths) > 0:
        max_start = int(np.sum(segment_lengths[:np.argmax(segment_lengths)])) # get the starting index of the maximum segment
        max_end = int(max_start+np.max(segment_lengths)) # get the ending index of the maximum segment
        zscore_mean = np.mean(z_score[max_start:max_end]) # get the z_score mean of that interval

    else: # if there is no maximum segment get the z-score means of the outliers
        zscore_mean = np.mean(z_score)

    return zscore_mean
