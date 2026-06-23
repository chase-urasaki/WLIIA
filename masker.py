#%% 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import peak_widths
#%% 

He_spec = pd.read_csv("wav_cal_optimal_red_nspec200714_0226_70.spectrum", delim_whitespace=True)

# %%
# remove 0 valued spectrum points 
He_spec_clean = He_spec[He_spec['Spectrum'] > 0]
# %%
plt.plot(He_spec_clean['Wavelength'], He_spec_clean['Spectrum'])
# %%
region1 = (np.min(He_spec_clean['Wavelength']), np.max(He_spec_clean['Wavelength']))
#%% 
plt.plot(He_spec_clean['Wavelength'], He_spec_clean['Spectrum'])
# %%
region2 = (1.0812,1.0847)
# %%
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
# %%
nist_df = clean_nist_csv('NIST_lines.csv')

line_dict = build_line_dict(nist_df)
# %%
line_dict
# %%
def match_known_lines(wavelengths, flux, line_dict, tolerance=0.01, prominence=0.2, region=None):
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
# %%
matched_lines = match_known_lines(He_spec_clean['Wavelength'].values, He_spec_clean['Spectrum'].values, line_dict, tolerance=0.01, prominence=0.16, region=region2)
# %%
matched_lines
# %%
# Plot the matched lines 
def plot_with_matches(wavelengths, flux, matches, region=None):
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, flux, label='Spectrum', color='black')
    
    # Define color mapping for elements
    color_map = {
        'Fe': 'orange', 
        'Na': 'blue', 
        'Mg': 'green', 
        'Ti': 'purple', 
        'He': 'red',
        'Ca': 'brown',
        'Si': 'pink'
    }
    default_color = 'gray'  # For unknown elements
    
    # Track which elements are used for the legend
    used_elements = set()
    
    for match in matches:
        if region and not (region[0] <= match['wavelength'] <= region[1]):
            continue
            
        element = match['element']
        color = color_map.get(element, default_color)
        used_elements.add(element)
        
        # Plot the line
        plt.axvline(match['wavelength'], color=color, linestyle='--', alpha=0.7, linewidth=1)
        
        # Add text label (optional - you can remove this if it gets cluttered)
        # plt.text(match['wavelength'], np.max(flux)*0.9, f"{match['reference']:.3f}", 
        #         rotation=90, verticalalignment='bottom', fontsize=6, color=color)
    
    # Create legend entries for each element used
    legend_elements = [plt.Line2D([0], [0], color='black', lw=2, label='Spectrum')]
    
    for element in sorted(used_elements):
        color = color_map.get(element, default_color)
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, 
                                        linestyle='--', label=f'{element} Lines'))
    
    plt.legend(handles=legend_elements, loc='best')
    
    if region:
        plt.xlim(region)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.title('Spectrum with Matched Lines (by Element)')
    plt.grid(True, alpha=0.3)
    plt.show()
# %%
plot_with_matches(He_spec_clean['Wavelength'].values, He_spec_clean['Spectrum'].values, matched_lines, region2)
# %%
# Build the mask 
def build_line_mask_FWHM(wavelengths, flux, matches, buffer=1.2):
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
            
    # self.exclude_regions = exclude_regions
    return mask, exclude_regions
# %%
mask, exclude_regions = build_line_mask_FWHM(He_spec_clean['Wavelength'].values, He_spec_clean['Spectrum'].values, matched_lines, buffer=1.1)
# %%
plt.figure(figsize=(12, 6))
plt.plot(He_spec_clean['Wavelength'].values, He_spec_clean['Spectrum'].values, label='Spectrum', color='black')
plt.plot(He_spec_clean['Wavelength'].values[mask], He_spec_clean['Spectrum'].values[mask], label='Masked Spectrum', color='red')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux')
plt.title('Spectrum with Masked Regions')
plt.legend()
if region2:
    plt.xlim(region2)
plt.grid(True, alpha=0.3)
# %%
exclude_regions

# %%
# Output for molecfit: 
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


# %%
print_molecfit_rc_lines(include_regions=[region2], exclude_regions=exclude_regions, precision=7)
# %%
