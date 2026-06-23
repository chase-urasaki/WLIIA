#%%
import numpy as np 
import pandas as pd
from scipy.signal import find_peaks
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import peak_widths

#%% 
# Make a function to read in your spectrum file 
def read_spectrum(file_path):
    """
    Reads in a spectrum file and returns the wavelength and flux arrays.

    Parameters:
    file_path (str): The path to the spectrum file.

    Returns:
    tuple: A tuple containing two numpy arrays: wavelength and flux.
    """
    # Read the FITS file
    with fits.open(file_path) as hdul:
        data = hdul[1].data  # Assuming the spectrum is in the first extension
        wavelength = data['wavelength']  # Replace with actual column name
        flux = data['flux']  # Replace with actual column name

    return wavelength, flux

def match_NIST_lines