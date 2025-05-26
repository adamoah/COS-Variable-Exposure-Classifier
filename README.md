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
-v and -vv indicates the level of verbosity and how much informaion is printed to the screen during execution
-i indicates the path to the input text containing the exposure file paths (An example can be found in example.txt)
-o indicates the output directory to store any saved figures, data array, or log files

You can also modify the following user defined variables in ```cos_exposure_finder.py``` to modify the code's behavior
'''

'''
