# import necessary modules (will add more as needed)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import scipy as sp
from scipy.ndimage import convolve1d
from scipy.signal.windows import gaussian

# astropy submodules
import astropy
from astropy.table import Table
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

# system pth to custom cos function file, time for time tracking
import sys
import time as t
sys.path.append("./custom_cos_functions")
# functions I created
from custom_cos_functions import *

# os, glob, and json for file manipulation
import os
import glob
import pickle



def main():
    # -v, -vv : verbose and very verbose prints information on what the program is doing
    # -i : input text file contains list of fits file paths to process
    # -o : output directory to dump saved data and/or plots

    args = sys.argv
    
    # command line variables + predefined variables
    total = 0
    verbose = 0
    fits_files = []
    error_files = ""
    badttimes = "NAME, START, END\n"
    outputdir = "./"
    lref = os.environ['lref']

    # user defined variables
    f_neighbors = 0.5 # number of neighbors to consider when running lof (as a fraction of the total length of the count rate arrays)
    cutoff = 0.05 # fraction cutoff for significant segment lengths
    save_data = True # save the count_rate + parameter data
    plot_cr = False # plots count rate data
    derivative = False # Find MJD start and stop time using the derivative method. If False using maximum segment method
    logerror = False # whether or not to log files that were unable to be parsed due to an error

    # intervals (not inclusive) to determine if an exposure plot should be saved
    max_interval = (-999, 999)
    occur_interval = (-999, 999)
    z_score_interval = (-999, 999)

    # data arrays
    names, means, maxes, occurs, count_arrs, = [], [], [], [], []
    for i in range(len(args)): # process each argument

        if args[i] == "-vv": # very verbose
            verbose = 2
        
        if args[i] == "-v": # verbosity
            verbose = 1
            
        elif args[i] == "-i": # read in text file
            i += 1
            with open(args[i], "r") as paths:
                fits_files = paths.read().split("\n") # reads text
                
        elif args[i] == "-o": # read in output dir
            i += 1
            # adds a slash if it is not already there
            outputdir = args[i] + "/" if args[-1] != "/" else args[i]
            os.makedirs(outputdir, exist_ok=True) # make the directory if it does not already exist

    # number of files found
    if verbose >= 1:
        print(f"number of files: {len(fits_files)}")


    # start processing
    for idx, file in enumerate(fits_files):

        if verbose >= 1:
            print(f"processing {file}")
        

        # extract data from the file
        try:
            data = extract_from_file(file, 1, method='TABLE')
            hdr0 = extract_from_file(file, 0, header=True)
            hdr1 = extract_from_file(file, 1, header=True) 
        except Exception as e:
            print(f"Error, file {file} could not be processed")
            if logerror: # log file that was unable to be processed
                error_files += file + " | " + e

        xtractab, grat, cw, aper, segment = hdr0['XTRACTAB'], hdr0['OPT_ELEM'], hdr0['CENWAVE'], hdr0['APERTURE'], (hdr0['SEGMENT'] if hdr0['SEGMENT'] != "N/A" else "NUV")

        if aper != "PSA": # filter for non PSA exposures
            if verbose >= 1:
                print(f"File: {file} cannot be processed because it is not PSA.")
            continue
        
        if xtractab == "N/A": # do not continue processing if the extract tab cannot be found
            if verbose >= 1:
                print(f"File {file} cannot be procesed because XTRACTAB is empty.")
            continue

        xtractab = lref + xtractab.split('$')[1]

        # get the PID and exposure id, lenth, and start time
        name, exptime, expstart = f"{hdr0['PROPOSID']}_{hdr1['EXPNAME'].strip()}_{segment}", hdr1['EXPTIME'], hdr1['EXPSTART']
        # extract data
        time, ycorr, wl = data['TIME'], data['YCORR'], data['WAVELENGTH']
        if (len(time) == 0): # filter for empty file or non PSA exposures
            if verbose >= 1:
                print(f"File: {file} cannot be processed because it is empty.")
            continue
        

        if segment == 'NUV':
            bkg_start = 3
            h_ratio = np.mean([hdr1['B_HGT1_A'], hdr1['B_HGT1_B'], hdr1['B_HGT1_C']]) / np.mean([hdr1['SP_HGT_A'], hdr1['SP_HGT_B'], hdr1['SP_HGT_C']])
        elif segment == 'FUVA':
            bkg_start = 1
            h_ratio = hdr1['B_HGT1_A'] / hdr1['SP_HGT_A']
        elif segment == 'FUVB':
            bkg_start = 1
            h_ratio =  hdr1['B_HGT1_B'] / hdr1['SP_HGT_B']
        elif segment == 'BOTH':
            bkg_start = 2
            h_ratio =  np.mean([hdr1['B_HGT1_A'], hdr1['B_HGT1_B']]) / np.mean([hdr1['SP_HGT_A'], hdr1['SP_HGT_B']])

        try:
            bounding_data, _ = readxtractab(xtractab, grat, cw, aper, segment) # read in bounding box info
        except FileNotFoundError:
            if verbose >= 1:
                print(f"File {file} cannot be procesed because XTRACTAB is empty.")
            continue

        bin_size = 0.16 if segment == 'NUV' else 0.64 # define bins

        # compute count rates and do backgroun subtraction
        counts, unique_timesteps = get_count_rate_vs_time(time, ycorr, wl, bounding_data[:bkg_start], exptime, bin_size=bin_size, verbose=verbose)
        bkg_counts, _ = get_count_rate_vs_time(time, ycorr, wl, bounding_data[bkg_start:], exptime, bin_size=bin_size, verbose=verbose)

        length = min(len(counts), len(bkg_counts)) - 1 # bound count rates by the min length of the two
        counts = counts[:length] - (bkg_counts[:length]*h_ratio) if (length > 0) else counts

        # do outlier prediction
        x = np.where(np.isnan(counts), 0, counts)
        if len(x) <= 1: # exposure must have at least two timesteps with counts
            continue
        n_neighbors = int(len(x) * f_neighbors)
        y = LocalOutlierFactor(n_neighbors=n_neighbors).fit_predict(x.reshape(-1, 1))

        # get anomalies and z score
        anomalies = x[y == -1]
        z_score = (anomalies - np.median(x)) / np.std(x)

        idx_arr = (y == -1).nonzero()[0] # get the index locations of all outliers
        
        # process if any outliers were found
        if len(idx_arr) > 0:
   
            segment_lengths, idx_positions = get_outlier_segments(idx_arr)

            if derivative: # get start and end using the derivative method
                start, end = get_derivative_positions(x)
            else: # else use maximum segment method
                start, end = idx_positions[np.argmax(segment_lengths)] if len(idx_positions) > 0 else (0, 0)
            
            # convert to MJD and append badttimes
            START = expstart + (unique_timesteps[start] / (3600*24))
            END = expstart + (unique_timesteps[end] / (3600*24))
            badttimes += f"{name},{START}, {END}\n"

            # calculate stats
            max_ = np.max(segment_lengths) # stats for segment lengths
            mask = (segment_lengths / len(anomalies) >= cutoff)
            occur = np.count_nonzero(mask)
            z_mean = get_max_zscore(z_score, segment_lengths)

            # save exposure info
            if save_data:
                count_arrs.append(x)
                means.append(z_mean)
                maxes.append(max_)
                occurs.append(occur)
                names.append(name)

            # only accept those of which which satistfy these conditions
            if (plot_cr and (max_ > max_interval[0] and max_ < max_interval[1]) and
                (occur > occur_interval[0] and occur < occur_interval[1]) and
                (z_mean > z_score_interval[0] and z_mean < z_score_interval[1])):

                
                # plots the raw count rates with highlighted outliers and parameter data
                fig, ax0 = plt.subplots(figsize=(13, 6))
                ax0.plot(unique_timesteps[:len(x)], x, color="#ff7f00", label=f"z-score: {z_mean}\nmax length: {max_}\nsegment count: {occur}")
                ax0.plot(unique_timesteps[:len(x)], np.where(y==1, x, None), color="#377eb8")

                # display badt start and end times
                ax0.axvline(unique_timesteps[start], color='red', label=f'Start:({START})')
                ax0.axvline(unique_timesteps[end], color='red', label=f'End ({END})')
        
                # plot labels 
                ax0.set_title(name + ' Count Rate')
                ax0.set_xlabel('seconds')
                ax0.set_ylabel('counts')
                ax0.legend()
        
                plt.close(fig)
                fig.savefig(outputdir + name + ".png")

        total += 1

    if (save_data): # save parameter and count rate data
        with open(outputdir + 'cos_files_data.pkl', 'wb') as Out:
            out_data = (names, means, maxes, occurs, count_arrs)
            pickle.dump(out_data, Out)
    
    if (logerror): # save error logged data
        with open(outputdir + 'log_err_files.txt', 'w') as Out:
            Out.write(error_files)
    
    with open(outputdir + 'badttimes.txt', 'w') as Out: # save badttimes
        Out.write(badttimes)

    if verbose >= 1:
        print(f"number of processed files: {(total)}")

if __name__ == "__main__":
    main()
