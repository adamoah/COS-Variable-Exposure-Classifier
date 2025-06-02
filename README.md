# COS-Variable-Exposure-Classifier

This repository includes code designed to find and classify variable exposures from the Cosmic Origins Spectrograph aboard the Hubble Space Telescope.
The code includes a number of options on how the exposure information is stored, saved, and displayed.

More information can be found in the ISR located here [PENDING]

Download the necessary libraries using
```
pip install -r requirements.txt
```

You can run the code by using the following terminal command
```
python ./cos_exposure_finder.py -v -i ./input_dir/input.txt -o ./output_dir
```
with the following command line arguments:
1. -v and -vv indicates the level of verbosity and how much informaion is printed to the screen during execution
2. -i indicates the path to the input text containing the exposure file paths (An example can be found in example.txt)
3. -o indicates the output directory to store any saved figures, data array, or log files

(Note: if you have an existing directory of exposure files and want to processing them using this code entering ```ls -d "$PWD"/* > ./input.txt``` to easily store the file paths in a text file.)

You can also modify the following user defined variables in ```cos_exposure_finder.py``` to modify the code's behavior
```
# user defined variables
f_neighbors = 0.5 # number of neighbors to consider when running lof (as a fraction of the total length of the count rate arrays)

cutoff = 0.05 # fraction cutoff of total number of outliers detected by lof for significant segment lengths

save_data = True # save the count_rate + parameter data for all exposures

plot_cr = True # plots count rate data

badttab = True # display and save bad time intervals for exposures within the parameter intervals

derivative = False # Find MJD start and stop time using the derivative method. If False using maximum segment method

logerror = False # whether or not to log files that were unable to be parsed due to an error


# parameter intervals (not inclusive) to determine if an exposure plot should be saved
max_interval = (-999, 999)

occur_interval = (-999, 999)

z_score_interval = (-999, 999)
```
