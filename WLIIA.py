#%%
import numpy as np 
import pandas as pd
from scipy.signal import find_peaks
import numpy as np

#%%

class spectra:
    def __init__(self, data):
        self.data = data

    def get_spectra(self):
        return self.data

    def get_spectra_by_index(self, index):
        return self.data[index]

    def get_spectra_by_name(self, name):
        return self.data[name]  
    

#%%
def load_spectra(file_path):
    """
    Load spectra data from a file.
    
    Parameters:
    file_path (str): Path to the file containing spectra data.
    
    Returns:
    spectra: An instance of the spectra class containing the loaded data.
    """
    # Assuming the file is in CSV format for this example
    data = pd.read_csv(file_path)
    
    # Convert the DataFrame to a dictionary for easier access
    spectra_data = {name: data[name].values for name in data.columns}
    
    return spectra(spectra_data)

#%%
# Load the NIST lines (all in vacuum)
def clean_nist_csv(file):
    df = pd.read_csv(file)

    # Strip equals signs and quotes
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('="', '').str.replace('"', '').str.strip()

    # Convert numeric columns
    df['obs_wl_vac(nm)'] = pd.to_numeric(df['obs_wl_vac(nm)'], errors='coerce') / 1000  # nm -> microns
    df['intens'] = pd.to_numeric(df['intens'], errors='coerce')

    # Drop rows with no wavelength
    df = df.dropna(subset=['obs_wl_vac(nm)'])
    return df

def build_line_dict(df, elements=['Na', 'Mg', 'Fe', 'Ti'], min_intensity=50):
    line_dict = {}
    for elem in elements:
        mask = (df['element'] == elem) & (df['intens'] >= min_intensity)
        lines = df[mask]['obs_wl_vac(nm)'].values
        line_dict[elem] = lines
    return line_dict


def match_lines(wavelengths, flux, line_dict, tolerance=0.1):
    inverted = 1 - flux
    peaks, _ = find_peaks(inverted, height=0.01)

    matched = []
    for idx in peaks:
        wl = wavelengths[idx]
        for elem, lines in line_dict.items():
            for ref in lines:
                if abs(wl - ref) < tolerance:
                    matched.append({
                        "element": elem,
                        "wavelength": wl,
                        "reference": ref,
                        "index": idx
                    })
    return matched

if __name__ == "__main__":
    # Read the NIST lines
    nist_file = 'nist_lines.csv'
    nist_df = clean_nist_csv(nist_file)
    line_dict = build_line_dict(nist_df)
    print("NIST lines loaded and cleaned.")

#%%