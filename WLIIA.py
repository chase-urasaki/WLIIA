# Whose Line Is It Anyway - Prototype Script
#%%
# Whose Line Is It Anyway - Prototype Script
#%%
import numpy as np 
import pandas as pd
from scipy.signal import find_peaks
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Spectrum:
    def __init__(self, data):
        self.data = data

    def get(self):
        return self.data

    def get_by_index(self, index):
        return self.data[index]

    def get_by_name(self, name):
        return self.data[name]

    def match_known_lines(self, line_dict, tolerance=0.01, prominence=0.2, region=None):
        wavelengths = self.data['wavelength']
        flux = self.data['flux']
        inverted = 1 - flux
        peaks, _ = find_peaks(inverted, height=prominence)

        matched = []
        for idx in peaks:
            wl = wavelengths[idx]
            if region and not (region[0] <= wl <= region[1]):
                continue

            closest = None
            min_diff = tolerance
            for elem, lines in line_dict.items():
                for ref in lines:
                    diff = abs(wl - ref)
                    if diff < min_diff:
                        closest = {"element": elem, "wavelength": wl, "reference": ref, "index": idx}
                        min_diff = diff
            if closest:
                matched.append(closest)
        return matched

    def plot_with_tellurics(self, matches=None, telluric=None, region=None):
        wl = self.data['wavelength']
        flux = self.data['flux']

        plt.figure(figsize=(12, 5))
        plt.plot(wl, flux, label='Spectrum')

        if telluric:
            tell_wl = telluric['wavelength']
            trans = telluric['transmission']
            plt.plot(tell_wl, trans, 'k--', label='Telluric Model')

        color_map = {
            'Fe': 'orange', 'Na': 'blue', 'Mg': 'green', 'Ti': 'purple', 'He': 'red'
        }
        used_elements = set()

        if matches:
            for match in matches:
                color = color_map.get(match['element'], 'gray')
                used_elements.add(match['element'])
                plt.axvline(match['wavelength'], color=color, linestyle='--', alpha=0.6)

        if region:
            plt.xlim(region)

        plt.ylim(flux.min() - 0.1, flux.max() + 0.1)
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Flux")

        # Add custom legend for matched lines
        legend_elements = [Line2D([0], [0], color=color_map[el], lw=2, linestyle='--', label=el)
                           for el in sorted(used_elements) if el in color_map]
        plt.legend(handles=[Line2D([0], [0], color='black', lw=2, label='Spectrum')] +
                           ([Line2D([0], [0], color='k', lw=2, linestyle='--', label='Telluric Model')] if telluric else []) +
                           legend_elements)

        plt.tight_layout()
        plt.show()

def load_spectrum(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
        spectra_data = {name: data[name].values for name in data.columns}
    elif file_path.endswith('.fits'):
        with fits.open(file_path) as hdul:
            table = hdul[1].data
            spectra_data = {
                'wavelength': np.array(table['wavelength']),
                'flux': np.array(table['flux'])
            }
    else:
        raise ValueError("Unsupported file format")
    return Spectrum(spectra_data)

def load_telluric_model(file_path):
    hdul = fits.open(file_path)
    # Find usable HDU
    for i in [1, 0]:
        data = hdul[i].data
        if data is not None:
            break

    # Extract wavelength and transmission
    if data.ndim == 1:
        transmission = data
        wavelength_tel = np.arange(len(data))
    elif data.ndim == 2:
        if data.shape[0] == 2:
            wavelength_tel = data[0]
            transmission = data[1]
        elif data.shape[1] == 2:
            wavelength_tel = data[:, 0]
            transmission = data[:, 1]
        else:
            raise ValueError("Unexpected shape in telluric model.")
    else:
        raise ValueError("Unsupported telluric data shape.")
    return {
        'wavelength': wavelength_tel, 
        'transmission': transmission
    }

def clean_nist_csv(file):
    df = pd.read_csv(file)
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('="', '').str.replace('"', '').str.strip()
    df['obs_wl_vac(nm)'] = pd.to_numeric(df['obs_wl_vac(nm)'], errors='coerce') / 1000
    df['intens'] = pd.to_numeric(df['intens'], errors='coerce')
    df = df.dropna(subset=['obs_wl_vac(nm)'])
    return df

def build_line_dict(df, elements=['Na', 'Mg', 'Fe', 'Ti'], min_intensity=0, top_n=None):
    line_dict = {}
    for elem in elements:
        mask = (df['element'] == elem) & (df['intens'] >= min_intensity)
        subset = df[mask].copy()

        if top_n is not None:
            subset = subset.sort_values(by='intens', ascending=False).head(top_n)

        line_dict[elem] = subset['obs_wl_vac(nm)'].values
    return line_dict

#%%
#%%
# Example usage (in another script or notebook):
region1 = (0.652328, 0.652846)
broadband = 0.649, 0.657
spectrum = load_spectrum("KP202401202105978_molecfit_norm_normalized.fits")
telluric = load_telluric_model("TELLURIC_CORR.fits")
nist_df = clean_nist_csv("NIST_lines.csv")
line_dict = build_line_dict(nist_df)
matches = spectrum.match_known_lines(line_dict, region=broadband, tolerance=0.01, prominence=0.2)
spectrum.plot_with_tellurics(matches=matches, telluric=telluric, region=broadband)
# The above code is a prototype for analyzing spectra and matching known lines.
#%%
matches
# %%
