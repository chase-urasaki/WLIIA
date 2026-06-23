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

    def build_line_mask_FWHM(self, matches, buffer=1.1):
        """
        Build a mask to exclude spectral lines based on their FWHM.
        
        Parameters
        ----------
        matches : pandas DataFrame or list of dict
            DataFrame containing matches with an 'index' column, or list of dictionaries
        buffer : float
            Multiplier to make the masked region wider than the FWHM
            
        Returns
        -------
        mask : ndarray
            Boolean mask where True indicates wavelength points to keep
        """
        wavelengths = self.data['wavelength']
        flux = self.data['flux']
        mask = np.ones(len(wavelengths), dtype=bool)
        exclude_regions = []

        # Handle matches as either DataFrame or list of dicts
        if isinstance(matches, pd.DataFrame):
            # Process DataFrame rows
            for _, row in matches.iterrows():
                if 'index' in row:
                    idx = row['index']
                else:
                    # If no index is provided but wavelength is, find nearest index
                    if 'wavelength' in row:
                        wl = row['wavelength']
                    elif 'observed_wl' in row:
                        wl = row['observed_wl']
                    else:
                        continue  # Skip if no wavelength information
                    
                    idx = np.argmin(np.abs(wavelengths - wl))
                
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
        else:
            # Process list of dictionaries (original implementation)
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

    def plot_mask(self, mask):
        wavelengths = self.data['wavelength']
        flux = self.data['flux']

        masked_flux = flux[mask]

        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, masked_flux)
    


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
        """
        Plot spectrum with matched lines, telluric model, and mask.
        
        Parameters
        ----------
        matches : list of dict, optional
            List of matched line dictionaries
        telluric : dict, optional
            Telluric model dictionary with 'transmission' key
        region : tuple, optional
            Wavelength region (min, max) to display
        mask : ndarray, optional
            Boolean mask for the spectrum
        """
        wl = self.data['wavelength']
        flux = self.data['flux']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(wl, flux, label='Original Spectrum', color='black', lw=1)

        if telluric:
            trans = telluric['transmission']
            ax1.plot(wl, trans, 'k--', label='Telluric Model', alpha=0.7)

        color_map = {
            'Fe': 'orange', 'Na': 'blue', 'Mg': 'green', 'Ti': 'purple', 
            'He': 'red', 'Ca': 'brown', 'Si': 'pink'
        }
        used_elements = set()

        if matches:
            for match in matches:
                color = color_map.get(match['element'], 'gray')
                used_elements.add(match['element'])
                ax1.axvline(match['wavelength'], color=color, linestyle='--', alpha=0.6, lw=0.8)

        if hasattr(self, 'exclude_regions'):
            for lower, upper in self.exclude_regions:
                ax1.axvspan(lower, upper, color='gray', alpha=0.3, label='Masked Region')
                ax2.axvspan(lower, upper, color='gray', alpha=0.3)

        ax1.set_ylabel("Flux")
        ax1.set_ylim(flux.min() - 0.1, flux.max() + 0.1)
        ax1.set_title("Original Spectrum with Matched Lines")
        ax1.grid(True, alpha=0.3)

        if mask is not None:
            ax2.plot(wl[mask], flux[mask], color='black', lw=1, label='Masked Spectrum')
            ax2.set_ylabel("Flux (Masked)")
            ax2.set_xlabel("Wavelength [μm]")
            ax2.set_ylim(flux.min() - 0.1, flux.max() + 0.1)
            ax2.set_title("Masked Spectrum (Lines Removed)")
            ax2.grid(True, alpha=0.3)
            if telluric:
                trans = telluric['transmission']
                ax2.plot(wl, trans, 'r--', label='Telluric Model', alpha=0.7)
            ax2.legend()

        if region:
            ax1.set_xlim(region)
            ax2.set_xlim(region)

        # Create legend for matched lines by element
        legend_elements = [Line2D([0], [0], color='black', lw=2, label='Spectrum')]
        if telluric:
            legend_elements.append(Line2D([0], [0], color='k', lw=2, linestyle='--', label='Telluric Model'))
        
        for el in sorted(used_elements):
            if el in color_map:
                legend_elements.append(Line2D([0], [0], color=color_map[el], lw=2, linestyle='--', label=f'{el} Lines'))
        
        if hasattr(self, 'exclude_regions') and len(self.exclude_regions) > 0:
            legend_elements.append(Line2D([0], [0], color='gray', lw=4, alpha=0.3, label='Masked Region'))
        
        ax1.legend(handles=legend_elements, loc='best')

        plt.tight_layout()
        plt.show()

def load_spectrum(file_path_or_obj):
    """
    Load spectral data from CSV, FITS, or .spectrum files.
    
    Parameters
    ----------
    file_path_or_obj : str or UploadedFile
        Either a file path string or a Streamlit UploadedFile object
        
    Returns
    -------
    Spectrum
        A Spectrum object containing the loaded data
    """
    # Handle Streamlit UploadedFile objects
    if hasattr(file_path_or_obj, 'name'):
        file_name = file_path_or_obj.name
        
        if file_name.endswith('.csv') or file_name.endswith('.spectrum'):
            data = pd.read_csv(file_path_or_obj, delim_whitespace=True)
            spectra_data = {name: data[name].values for name in data.columns}
        elif file_name.endswith('.fits'):
            with fits.open(file_path_or_obj) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        if isinstance(hdu.data, np.ndarray) and len(hdu.data.shape) == 1:
                            table = hdu.data
                            # Try different column name combinations
                            col_names = table.dtype.names
                            
                            # Look for wavelength column (various names)
                            wl_col = None
                            for name in ['wavelength', 'Wavelength', 'WAVELENGTH', 'wave', 'Wave', 'WAVE']:
                                if name in col_names:
                                    wl_col = name
                                    break
                            
                            # Look for flux column (various names)
                            flux_col = None
                            for name in ['flux', 'Flux', 'FLUX', 'spectrum', 'Spectrum', 'SPECTRUM']:
                                if name in col_names:
                                    flux_col = name
                                    break
                            
                            if wl_col and flux_col:
                                # Convert wavelength from Angstroms to microns if needed
                                wl_data = np.array(table[wl_col])
                                if wl_data.max() > 100:  # Likely in Angstroms
                                    wl_data = wl_data / 10000.0  # Convert to microns
                                
                                spectra_data = {
                                    'wavelength': wl_data,
                                    'flux': np.array(table[flux_col])
                                }
                                break
                else:
                    raise ValueError(f"Could not find valid wavelength and flux data in FITS file: {file_name}")
        else:
            raise ValueError(f"Unsupported file format: {file_name}")
    
    # Handle string file paths
    elif isinstance(file_path_or_obj, str):
        file_path = file_path_or_obj
        
        if file_path.endswith('.csv') or file_path.endswith('.spectrum'):
            data = pd.read_csv(file_path, delim_whitespace=True)
            spectra_data = {name: data[name].values for name in data.columns}
        elif file_path.endswith('.fits'):
            with fits.open(file_path) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        if isinstance(hdu.data, np.ndarray) and len(hdu.data.shape) == 1:
                            table = hdu.data
                            col_names = table.dtype.names
                            
                            # Look for wavelength column
                            wl_col = None
                            for name in ['wavelength', 'Wavelength', 'WAVELENGTH', 'wave', 'Wave', 'WAVE']:
                                if name in col_names:
                                    wl_col = name
                                    break
                            
                            # Look for flux column
                            flux_col = None
                            for name in ['flux', 'Flux', 'FLUX', 'spectrum', 'Spectrum', 'SPECTRUM']:
                                if name in col_names:
                                    flux_col = name
                                    break
                            
                            if wl_col and flux_col:
                                # Convert wavelength from Angstroms to microns if needed
                                wl_data = np.array(table[wl_col])
                                if wl_data.max() > 100:  # Likely in Angstroms
                                    wl_data = wl_data / 10000.0  # Convert to microns
                                
                                spectra_data = {
                                    'wavelength': wl_data,
                                    'flux': np.array(table[flux_col])
                                }
                                break
                else:
                    raise ValueError(f"Could not find valid wavelength and flux data in FITS file: {file_path}")
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    else:
        raise TypeError("Expected string file path or UploadedFile object")
    
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
# # Run ESPRESSO line matching on the entire order
# order = (0.649, 0.657)
# spectrum = load_spectrum("KP202401202105978_molecfit_norm_normalized.fits")
# telluric = load_telluric_model("TELLURIC_CORR.fits")

# # Load ESPRESSO line list
# espresso_lines = load_espresso_lines("K6_espresso.txt")  # Replace with your actual file

# # Match lines across the entire order
# order_espresso_matches = spectrum.match_espresso_lines(
#     espresso_lines, 
#     region=order,
#     tolerance=0.0005,  # Tighter tolerance for ESPRESSO lines (in μm)
#     prominence=0.1
# )

# # Create a mask based on the matched lines
# order_mask = spectrum.build_line_mask_EW(order_espresso_matches, buffer=1.2)

# # Modify plot_with_tellurics function call to handle ESPRESSO matches
# spectrum.plot_with_tellurics(matches=order_espresso_matches, telluric=telluric, region=order, mask=order_mask)

# # Get Molecfit RC lines for the entire order
# print_molecfit_rc_lines(include_regions=[order], exclude_regions=spectrum.exclude_regions, precision=7)
# #%%
# # Example usage (in another script or notebook):
# region1 = (0.651, 0.652)
# order = (0.649, 0.657)
# spectrum = load_spectrum("KP202401202105978_molecfit_norm_normalized.fits")
# telluric = load_telluric_model("TELLURIC_CORR.fits")
# #nist_df = clean_nist_csv("NIST_lines.csv")
# #ine_dict = build_line_dict(nist_df)

# #order_matches = spectrum.match_known_lines(line_dict, region=order, tolerance=0.01, prominence=0.1)
# order_mask = spectrum.build_line_mask_FWHM(order_matches, buffer=1.1)
# spectrum.plot_with_tellurics(matches=order_matches, telluric=telluric, region=order, mask = order_mask)

# # The above code is a prototype for analyzing spectra and matching known lines.
# #%%
# region1 = (0.651, 0.652)
# region1_matches = spectrum.match_known_lines(line_dict, region=region1, tolerance=0.01, prominence=0.15)
# region1_mask = spectrum.build_line_mask_FWHM(region1_matches, buffer=1.2)
# spectrum.plot_with_tellurics(matches=region1_matches, telluric=telluric, region=region1, mask = region1_mask)

# # return molecfit rc bounds
# print_molecfit_rc_lines(include_regions=[region1], exclude_regions=spectrum.exclude_regions, precision=7)
# # %%
# region2 = (0.654, 0.65575)
# region2_matches = spectrum.match_known_lines(line_dict, region=region2, tolerance=0.01, prominence=0.15)
# region2_mask = spectrum.build_line_mask_FWHM(region2_matches, buffer=1.2)
# spectrum.plot_with_tellurics(matches=region2_matches, telluric=telluric, region=region2, mask = region2_mask)
# # %%
# print_molecfit_rc_lines(include_regions=[region2], exclude_regions=spectrum.exclude_regions, precision=7)
# # %%
#%%
#%%
# Template code for opening and exploring FITS table data
from astropy.io import fits
import numpy as np
import pandas as pd

#%%
# Open the FITS file
fits_file = "KP.20241206.28831.52_order8.fits"
hdul = fits.open(fits_file)

#%%
# Print info about all HDUs (Header Data Units)
hdul.info()

#%%
# Access a specific HDU (usually data is in HDU 1)
hdu = hdul[1]  # Change index as needed

# Print header information
print(hdu.header)

#%%
# Access the data
data = hdu.data

# Check data type and shape
print(f"Data type: {type(data)}")
print(f"Data dtype: {data.dtype}")
print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

#%%
# For table data, print column names
if hasattr(data, 'dtype') and hasattr(data.dtype, 'names'):
    print("Column names:")
    print(data.dtype.names)

#%%
# Access specific columns from table
if hasattr(data, 'dtype') and data.dtype.names:
    # Example: access wavelength and flux columns
    # Adjust column names based on what you see above
    for col in data.dtype.names:
        print(f"\n{col}:")
        print(f"  Shape: {data[col].shape}")
        print(f"  Min: {np.min(data[col])}")
        print(f"  Max: {np.max(data[col])}")
        print(f"  Sample values: {data[col][:5]}")

#%%
# Convert to pandas DataFrame for easier viewing
if hasattr(data, 'dtype') and data.dtype.names:
    df = pd.DataFrame({name: data[name] for name in data.dtype.names})
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")

#%%
# Plot the data (if wavelength and flux columns exist)
import matplotlib.pyplot as plt

# Adjust column names based on your FITS file
if 'wavelength' in data.dtype.names and 'flux' in data.dtype.names:
    wavelength = data['wavelength']
    flux = data['flux']
    
    plt.figure(figsize=(12, 6))
    plt.plot(wavelength, flux)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title(f'Spectrum from {fits_file}')
    plt.grid(True, alpha=0.3)
    plt.show()

#%%
# Convert wavelength from angstroms to nm 
He_spec_clean = pd.DataFrame({
    'Wavelength': data['wavelength'] / 10,  # Convert Å to nm
    'Spectrum': data['spectrum']
})
#%%
# Close the FITS file when done
hdul.close()

#%%
# Alternative: Use context manager (automatically closes file)
with fits.open(fits_file) as hdul:
    hdul.info()
    data = hdul[1].data
    # Do your analysis here
    print(data.dtype.names if hasattr(data.dtype, 'names') else data)

#%%
# ============================================================================
# KPF DATA ANALYSIS - Complete workflow
# ============================================================================

#%%
# Load KP spectrum from FITS file
kp_spectrum = load_spectrum("KP.20241206.28831.52_order8.fits")

#%%
# Quick check of the loaded spectrum
print(f"Wavelength range: {kp_spectrum.data['wavelength'].min():.4f} to {kp_spectrum.data['wavelength'].max():.4f} μm")
print(f"Number of data points: {len(kp_spectrum.data['wavelength'])}")

#%%
# Plot the raw spectrum to inspect it
plt.figure(figsize=(12, 6))
plt.plot(kp_spectrum.data['wavelength'], kp_spectrum.data['flux'])
plt.xlabel('Wavelength (μm)')
plt.ylabel('Flux')
plt.title('KPF Spectrum - Order 8')
plt.grid(True, alpha=0.3)
plt.show()

#%%
# Build the NIST line dictionary
nist_df = clean_nist_csv("NIST_lines.csv")
line_dict = build_line_dict(nist_df, elements=['Fe', 'Na', 'Mg', 'Ti', 'He', 'Ca', 'Si'], top_n=100)

print(f"Loaded lines for elements: {list(line_dict.keys())}")
for elem, lines in line_dict.items():
    print(f"  {elem}: {len(lines)} lines")

#%%
# Define the wavelength region to analyze
wl_min = kp_spectrum.data['wavelength'].min()
wl_max = kp_spectrum.data['wavelength'].max()
region = (wl_min, wl_max)

print(wl_min, wl_max)

print(f"Analyzing region: {region[0]:.4f} to {region[1]:.4f} μm")

#%%
# Match known lines in the spectrum
# Adjust tolerance and prominence based on your data
kp_matches = kp_spectrum.match_known_lines(
    line_dict, 
    region=region, 
    tolerance=0.0005,    # 0.5 nm tolerance (adjust as needed)
    prominence=0.05      # Minimum line depth (adjust as needed)
)

print(f"Found {len(kp_matches)} matched lines")

#%%
# Display the matches as a DataFrame
if len(kp_matches) > 0:
    matches_df = pd.DataFrame(kp_matches)
    print("\nFirst 10 matches:")
    print(matches_df.head(10))
    
    # Show breakdown by element
    print("\nMatches by element:")
    print(matches_df['element'].value_counts())
else:
    print("No matches found - try adjusting tolerance or prominence parameters")

#%%
# Build a mask using FWHM method
if len(kp_matches) > 0:
    kp_mask = kp_spectrum.build_line_mask_FWHM(kp_matches, buffer=1.2)
    print(f"Masked {np.sum(~kp_mask)} pixels out of {len(kp_mask)} total")
else:
    kp_mask = None

#%%
# Plot spectrum with matched lines (without telluric for now)
if len(kp_matches) > 0:
    kp_spectrum.plot_with_tellurics(
        matches=kp_matches, 
        telluric=None,  # Set to None if you don't have telluric data
        region=region, 
        mask=kp_mask
    )

#%%
# Generate Molecfit configuration for wavelength exclusion
if len(kp_matches) > 0 and hasattr(kp_spectrum, 'exclude_regions'):
    print("\nMolecfit wavelength configuration:")
    print_molecfit_rc_lines(
        include_regions=[region], 
        exclude_regions=kp_spectrum.exclude_regions, 
        precision=7
    )

#%%
# ============================================================================
# H-ALPHA LINE MASKING
# ============================================================================

#%%
# Define H-alpha line parameters
h_alpha_center = 0.65628  # H-alpha wavelength in microns (656.28 nm)
h_alpha_width = 0.0005    # Half-width to mask around H-alpha (adjustable)

# Check if H-alpha is in the wavelength range of your spectrum
wl_min = kp_spectrum.data['wavelength'].min()
wl_max = kp_spectrum.data['wavelength'].max()

if wl_min <= h_alpha_center <= wl_max:
    print(f"H-alpha line (0.65628 μm) is within the spectrum range")
    print(f"Spectrum covers: {wl_min:.5f} to {wl_max:.5f} μm")
    
    # Create H-alpha mask manually
    wavelengths = kp_spectrum.data['wavelength']
    h_alpha_mask = (wavelengths < h_alpha_center - h_alpha_width) | (wavelengths > h_alpha_center + h_alpha_width)
    
    # Create an exclude region for H-alpha
    h_alpha_exclude = [(h_alpha_center - h_alpha_width, h_alpha_center + h_alpha_width)]
    
    print(f"\nH-alpha mask excludes: {h_alpha_center - h_alpha_width:.5f} to {h_alpha_center + h_alpha_width:.5f} μm")
    print(f"Masked {np.sum(~h_alpha_mask)} pixels for H-alpha")
    
else:
    print(f"H-alpha line (0.65628 μm) is NOT in the spectrum range")
    print(f"Spectrum covers: {wl_min:.5f} to {wl_max:.5f} μm")
    h_alpha_mask = None
    h_alpha_exclude = []

#%%
# Combine the stellar line mask with H-alpha mask if both exist
if kp_mask is not None and h_alpha_mask is not None:
    # Combined mask (True = keep, False = exclude)
    combined_mask = kp_mask & h_alpha_mask
    
    # Combined exclude regions
    combined_exclude = kp_spectrum.exclude_regions + h_alpha_exclude
    
    print(f"\nCombined masking:")
    print(f"  Stellar lines masked: {np.sum(~kp_mask)} pixels")
    print(f"  H-alpha masked: {np.sum(~h_alpha_mask)} pixels")
    print(f"  Total masked: {np.sum(~combined_mask)} pixels")
    print(f"  Total exclude regions: {len(combined_exclude)}")
    
    # Store the combined exclude regions
    kp_spectrum.exclude_regions = combined_exclude
    
    # Plot with combined mask
    kp_spectrum.plot_with_tellurics(
        matches=kp_matches,
        telluric=None,
        region=region,
        mask=combined_mask
    )
    
    # Also create a zoomed plot around H-alpha
    h_alpha_region = (h_alpha_center - 0.002, h_alpha_center + 0.002)
    
    # Get matches in H-alpha region
    h_alpha_region_matches = [m for m in kp_matches 
                              if h_alpha_region[0] <= m['wavelength'] <= h_alpha_region[1]]
    
    print(f"\nFound {len(h_alpha_region_matches)} stellar lines near H-alpha")
    
    kp_spectrum.plot_with_tellurics(
        matches=h_alpha_region_matches,
        telluric=None,
        region=h_alpha_region,
        mask=combined_mask
    )
    
elif kp_mask is not None:
    combined_mask = kp_mask
    combined_exclude = kp_spectrum.exclude_regions
    print("Only stellar line mask available (H-alpha not in range)")
    
elif h_alpha_mask is not None:
    combined_mask = h_alpha_mask
    combined_exclude = h_alpha_exclude
    kp_spectrum.exclude_regions = combined_exclude
    print("Only H-alpha mask available (no stellar lines matched)")
    
else:
    combined_mask = None
    combined_exclude = []
    print("No masks available")

#%%
# Generate Molecfit configuration with H-alpha included
if combined_mask is not None and hasattr(kp_spectrum, 'exclude_regions'):
    print("\nMolecfit wavelength configuration (including H-alpha):")
    print_molecfit_rc_lines(
        include_regions=[region], 
        exclude_regions=kp_spectrum.exclude_regions, 
        precision=7
    )

#%%
# Plot whole spectrum with combined mask
if combined_mask is not None:
    kp_spectrum.plot_with_tellurics(
        matches=kp_matches,
        telluric=None,
        region=region,
        mask=combined_mask
    )

#%%
# ============================================================================
# SUB-REGION ANALYSIS FOR MOLECFIT
# ============================================================================

#%%
# Define your subregion
subregion = (0.650, 0.655)

# Filter exclude_regions to only those within the subregion
if hasattr(kp_spectrum, 'exclude_regions'):
    # Filter regions that overlap with the subregion
    subregion_exclude = []
    
    for lower, upper in kp_spectrum.exclude_regions:
        # Check if this exclude region overlaps with the subregion
        if lower <= subregion[1] and upper >= subregion[0]:
            # Clip the exclude region to fit within the subregion boundaries
            clipped_lower = max(lower, subregion[0])
            clipped_upper = min(upper, subregion[1])
            subregion_exclude.append((clipped_lower, clipped_upper))
    
    print(f"Subregion: {subregion[0]:.5f} to {subregion[1]:.5f} μm")
    print(f"Total exclude regions in full spectrum: {len(kp_spectrum.exclude_regions)}")
    print(f"Exclude regions within subregion: {len(subregion_exclude)}")
    
    # Create a mask specifically for the subregion
    wavelengths = kp_spectrum.data['wavelength']
    subregion_mask = np.ones(len(wavelengths), dtype=bool)
    
    for lower, upper in subregion_exclude:
        subregion_mask &= (wavelengths < lower) | (wavelengths > upper)
    
    # Also filter matches to only those in the subregion
    subregion_matches = [m for m in kp_matches 
                         if subregion[0] <= m['wavelength'] <= subregion[1]]


    print(f"Matched lines in subregion: {len(subregion_matches)}")
    
    # Temporarily store the filtered exclude regions
    original_exclude = kp_spectrum.exclude_regions
    kp_spectrum.exclude_regions = subregion_exclude[::-1]
    
    # Plot the subregion
    kp_spectrum.plot_with_tellurics(
        matches=subregion_matches,
        telluric=None,
        region=subregion,
        mask=subregion_mask
    )
    
    # Print detailed info about excluded regions
    print("\nExcluded regions in subregion:")
    for i, (lower, upper) in enumerate(subregion_exclude[::-1], 1):
        width_nm = (upper - lower) * 1000
        print(f"  {i:2d}. {lower:.7f} - {upper:.7f} μm  (width: {width_nm:.3f} nm)")
    
    # Generate Molecfit configuration for the subregion
    print("\n" + "="*70)
    print("MOLECFIT CONFIGURATION FOR SUBREGION")
    print("="*70)
    print_molecfit_rc_lines(
        include_regions=[subregion], 
        exclude_regions=subregion_exclude[::-1], 
        precision=7
    )
    print("="*70)
    
    # Optional: Save to file
    output_file = f"molecfit_config_subregion_{subregion[0]:.4f}_{subregion[1]:.4f}.txt"
    with open(output_file, 'w') as f:
        f.write("# Molecfit Configuration\n")
        f.write(f"# Subregion: {subregion[0]:.7f} to {subregion[1]:.7f} μm\n")
        f.write(f"# Number of excluded regions: {len(subregion_exclude[::-1])}\n")
        f.write(f"# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # WAVE_INCLUDE
        include_str = f"{subregion[0]:.7f},{subregion[1]:.7f}"
        f.write(f"WAVE_INCLUDE = {include_str}\n\n")
        
        # WAVE_EXCLUDE
        if subregion_exclude:
            exclude_flat = [val for pair in subregion_exclude[::-1] for val in pair]
            exclude_str = ",".join(f"{val:.7f}" for val in exclude_flat)
            f.write(f"WAVE_EXCLUDE = {exclude_str}\n\n")
        
        # Detailed list
        f.write("\n# Detailed list of excluded regions:\n")
        for i, (lower, upper) in enumerate(subregion_exclude[::-1], 1):
            width_nm = (upper - lower) * 1000
            f.write(f"# {i:2d}. {lower:.7f} - {upper:.7f} μm  (width: {width_nm:.3f} nm)\n")
    
    print(f"\nConfiguration saved to: {output_file}")
    
    # Restore original exclude regions
    kp_spectrum.exclude_regions = original_exclude

#%%