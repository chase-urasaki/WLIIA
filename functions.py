#%%
import numpy as np 
import pandas as pd
from scipy.signal import find_peaks
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import peak_widths

#%% 

def load_and_clean_nist_csv(file = "./data/NIST_lines.csv"):
    df = pd.read_csv(file)
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('="', '').str.replace('"', '').str.strip()
    df['obs_wl_vac(nm)'] = pd.to_numeric(df['obs_wl_vac(nm)'], errors='coerce') / 1000
    df['intens'] = pd.to_numeric(df['intens'], errors='coerce')
    df = df.dropna(subset=['obs_wl_vac(nm)'])
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

    def match_NIST_lines(self, line_dict, tolerance=0.01, prominence=0.2, region=None):
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

def load_spectrum(file_path_or_obj):
    """
    Load spectral data from CSV, FITS, or .spectrum files.
    
    Parameters
    ----------
    file_path_or_obj : str or UploadedFile
        Either a file path string or a Streamlit UploadedFile object
        
    Returns
    -------
    dict
        A dictionary containing the loaded spectral data
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