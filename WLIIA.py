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
from scipy.signal import peak_widths

def load_espresso_lines(file_path):
    """
    Load a two-column ESPRESSO line list: wavelength (Å) and EW.
    """
    df = pd.read_csv(file_path, delim_whitespace=True, names=["wavelength", "ew"], comment="#")
    return df

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
    
    def match_espresso_lines(self, line_df, tolerance=0.1, prominence=0.01, region=None):
        """
        Match observed spectrum lines to ESPRESSO-derived line list.

        Parameters:
            line_df : pandas DataFrame with 'wavelength' and 'ew' columns
            tolerance : float — matching tolerance in Å
            prominence : float — minimum line depth (used in peak finding)
            region : tuple(float, float) — optional wavelength region (start, end)

        Returns:
            List of dicts with matched observed and known line data
        """
        wavelengths = self.data['wavelength']
        flux = self.data['flux']
        inverted_flux = 1 - flux
        peaks, _ = find_peaks(inverted_flux, height=prominence)

        matched = []
        for idx in peaks:
            wl_obs = wavelengths[idx]
            if region and not (region[0] <= wl_obs <= region[1]):
                continue

            nearby_lines = line_df[np.abs(line_df['wavelength'] - wl_obs) <= tolerance]
            if not nearby_lines.empty:
                best_match = nearby_lines.iloc[nearby_lines['ew'].argmax()]
                matched.append({
                    'observed_wl': wl_obs,
                    'catalog_wl': best_match['wavelength'],
                    'catalog_ew': best_match['ew'],
                    'index': idx
                })

        return matched

    def build_line_mask_FWHM(self, matches, buffer = 1.1):
        wavelengths = self.data['wavelength']
        flux = self.data['flux']
        mask = np.ones(len(wavelengths), dtype=bool)
        exclude_regions = []

        for match in matches:
            idx = match['index']
            # Estimate FWHM using the peak width in index space
            region_flux = 1 - flux
            results_half = peak_widths(region_flux, [idx], rel_height=0.5)
            fwhm_pixels = results_half[0][0]

            dlambda = np.gradient(wavelengths)
            fwhm_lambda = fwhm_pixels * dlambda[idx]
            width = buffer * fwhm_lambda
            center = wavelengths[idx]
            lower = center - width
            upper = center + width
            exclude_regions.append((lower, upper))

            mask &= (wavelengths < center - width) | (wavelengths > center + width)
        self.exclude_regions = exclude_regions
        return mask

    def build_line_mask_EW(self, matches, buffer=1.5):
        """
        Build a mask to exclude spectral lines based on their equivalent width (EW).
        
        Parameters
        ----------
        matches : list of dict
            List of dictionaries containing matches, each with 'index' key
        buffer : float
            Multiplier to make the masked region wider than the calculated EW
            
        Returns
        -------
        mask : ndarray
            Boolean mask where True indicates wavelength points to keep
        """
        wavelengths = self.data['wavelength']
        flux = self.data['flux']
        mask = np.ones(len(wavelengths), dtype=bool)
        exclude_regions = []

        for match in matches:
            idx = match['index']
            
            # Find the local line boundaries by walking outward from the line center
            # until we reach close to the continuum level (using 98% of continuum)
            i_left = idx
            i_right = idx
            # Define continuum level as 1.0 (for normalized spectra)
            continuum = 1.0
            threshold = 0.98  # Consider 98% of continuum to be the line boundary
            
            while i_left > 0 and flux[i_left] < threshold * continuum:
                i_left -= 1
                
            while i_right < len(flux) - 1 and flux[i_right] < threshold * continuum:
                i_right += 1
            
            # Calculate EW by integrating over the line
            line_width_indices = i_right - i_left
            if line_width_indices <= 0:
                # Fallback if boundaries weren't found properly
                width = 0.0005  # Default small width in μm
            else:
                # Calculate EW in wavelength units
                wl_left = wavelengths[i_left]
                wl_right = wavelengths[i_right]
                width = (wl_right - wl_left) / 2  # Half-width
            
            # Apply buffer to width
            buffered_width = buffer * width
            center = wavelengths[idx]
            lower = center - buffered_width
            upper = center + buffered_width
            exclude_regions.append((lower, upper))
            
            # Update mask
            mask &= (wavelengths < lower) | (wavelengths > upper)
        
        self.exclude_regions = exclude_regions
        return mask

    def plot_with_tellurics(self, matches=None, telluric=None, region=None, mask=None):
            wl = self.data['wavelength']
            flux = self.data['flux']

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax1.plot(wl, flux, label='Original Spectrum')

            if telluric:
                trans = telluric['transmission']
                ax1.plot(wl, trans, 'k--', label='Telluric Model')

            color_map = {
                'Fe': 'orange', 'Na': 'blue', 'Mg': 'green', 'Ti': 'purple', 'He': 'red'
            }
            used_elements = set()

            if matches:
                for match in matches:
                    color = color_map.get(match['element'], 'gray')
                    used_elements.add(match['element'])
                    ax1.axvline(match['wavelength'], color=color, linestyle='--', alpha=0.6)

            if hasattr(self, 'exclude_regions'):
                for lower, upper in self.exclude_regions:
                    ax1.axvspan(lower, upper, color='gray', alpha=0.7, label='Masked Region')
                    ax2.axvspan(lower, upper, color='gray', alpha=0.7)

            ax1.set_ylabel("Flux")
            ax1.set_ylim(flux.min() - 0.1, flux.max() + 0.1)
            ax1.set_title("Original Spectrum with Tellurics and Matched Lines")

            if mask is not None:
                ax2.plot(wl[mask], flux[mask], color='black', lw=1, label='Masked Spectrum')
                ax2.set_ylabel("Flux (Masked)")
                ax2.set_xlabel("Wavelength [μm]")
                ax2.set_ylim(flux.min() - 0.1, flux.max() + 0.1)
                ax2.set_title("Masked Spectrum (Lines Removed)")
                ax2.plot(wl, trans, 'r--', label='Telluric Model')
                ax2.legend()

            if region:
                ax1.set_xlim(region)
                ax2.set_xlim(region)

            legend_elements = [Line2D([0], [0], color=color_map[el], lw=2, linestyle='--', label=el)
                            for el in sorted(used_elements) if el in color_map]
            ax1.legend(handles=[Line2D([0], [0], color='black', lw=2, label='Spectrum')] +
                            ([Line2D([0], [0], color='k', lw=2, linestyle='--', label='Telluric Model')] if telluric else []) +
                            legend_elements +
                            [Line2D([0], [0], color='gray', lw=4, alpha=0.3, label='Masked Region')])

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
    transmission = hdul[1].data

    return {
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

def print_molecfit_rc_lines(include_regions=None, exclude_regions=None, precision=7):
    """
    Print wl_include and wl_exclude in Molecfit-compatible .rc syntax.

    Parameters
    ----------
    include_regions : list of tuple
        List of (lower, upper) wavelength pairs for inclusion.
    exclude_regions : list of tuple
        List of (lower, upper) wavelength pairs for exclusion.
    precision : int
        Number of decimal places to include.
    """
    fmt = f"{{:.{precision}f}}"

    if include_regions:
        include_flat = [val for pair in include_regions for val in pair]
        include_str = ",".join(fmt.format(val) for val in include_flat)
        print(f"WAVE_INCLUDE = {include_str}")

    if exclude_regions:
        exclude_flat = [val for pair in exclude_regions for val in pair]
        exclude_str = ",".join(fmt.format(val) for val in exclude_flat)
        print(f"WAVE_EXCLUDE = {exclude_str}")

#%%
# Run this the espresso line matching on the order
# Load your spectrum from FITS or array
#%%
# Run ESPRESSO line matching on the entire order
order = (0.649, 0.657)
spectrum = load_spectrum("KP202401202105978_molecfit_norm_normalized.fits")
telluric = load_telluric_model("TELLURIC_CORR.fits")

# Load ESPRESSO line list
espresso_lines = load_espresso_lines("K6_espresso.txt")  # Replace with your actual file

# Match lines across the entire order
order_espresso_matches = spectrum.match_espresso_lines(
    espresso_lines, 
    region=order,
    tolerance=0.0005,  # Tighter tolerance for ESPRESSO lines (in μm)
    prominence=0.1
)

# Create a mask based on the matched lines
order_mask = spectrum.build_line_mask_EW(order_espresso_matches, buffer=1.2)

# Modify plot_with_tellurics function call to handle ESPRESSO matches
spectrum.plot_with_tellurics(matches=order_espresso_matches, telluric=telluric, region=order, mask=order_mask)

# Get Molecfit RC lines for the entire order
print_molecfit_rc_lines(include_regions=[order], exclude_regions=spectrum.exclude_regions, precision=7)
#%%
# Example usage (in another script or notebook):
region1 = (0.651, 0.652)
order = (0.649, 0.657)
spectrum = load_spectrum("KP202401202105978_molecfit_norm_normalized.fits")
telluric = load_telluric_model("TELLURIC_CORR.fits")
#nist_df = clean_nist_csv("NIST_lines.csv")
#ine_dict = build_line_dict(nist_df)

#order_matches = spectrum.match_known_lines(line_dict, region=order, tolerance=0.01, prominence=0.1)
order_mask = spectrum.build_line_mask_FWHM(order_matches, buffer=1.1)
spectrum.plot_with_tellurics(matches=order_matches, telluric=telluric, region=order, mask = order_mask)

# The above code is a prototype for analyzing spectra and matching known lines.
#%%
region1 = (0.651, 0.652)
region1_matches = spectrum.match_known_lines(line_dict, region=region1, tolerance=0.01, prominence=0.15)
region1_mask = spectrum.build_line_mask_FWHM(region1_matches, buffer=1.2)
spectrum.plot_with_tellurics(matches=region1_matches, telluric=telluric, region=region1, mask = region1_mask)

# return molecfit rc bounds
print_molecfit_rc_lines(include_regions=[region1], exclude_regions=spectrum.exclude_regions, precision=7)
# %%
region2 = (0.654, 0.65575)
region2_matches = spectrum.match_known_lines(line_dict, region=region2, tolerance=0.01, prominence=0.15)
region2_mask = spectrum.build_line_mask_FWHM(region2_matches, buffer=1.2)
spectrum.plot_with_tellurics(matches=region2_matches, telluric=telluric, region=region2, mask = region2_mask)
# %%
print_molecfit_rc_lines(include_regions=[region2], exclude_regions=spectrum.exclude_regions, precision=7)
# %%
