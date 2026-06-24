#%% 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from matplotlib.lines import Line2D

#%%
class Spectrum:
    """Class to handle spectrum loading, line matching, and masking operations."""
    
    def __init__(self, file_path=None, wavelengths=None, flux=None):
        """
        Initialize Spectrum from file or arrays.
        
        Parameters
        ----------
        file_path : str, optional
            Path to spectrum file
        wavelengths : array-like, optional
            Wavelength array
        flux : array-like, optional
            Flux array
        """
        if file_path is not None:
            data = pd.read_csv(file_path, delim_whitespace=True)
            # Remove 0 valued spectrum points
            data_clean = data[data['Spectrum'] > 0]
            self.wavelengths = data_clean['Wavelength'].values
            self.flux = data_clean['Spectrum'].values
        elif wavelengths is not None and flux is not None:
            self.wavelengths = np.asarray(wavelengths)
            self.flux = np.asarray(flux)
        else:
            raise ValueError("Must provide either file_path or both wavelengths and flux")
        
        self.matched_lines = None
        self.mask = None
        self.exclude_regions = []
    
    @property
    def wavelength_range(self):
        """Get the full wavelength range as a tuple."""
        return (np.min(self.wavelengths), np.max(self.wavelengths))
    
    @staticmethod
    def load_nist_data(file='./data/NIST_lines.csv'):
        """
        Load and clean NIST line database.
        
        Parameters
        ----------
        file : str
            Path to NIST CSV file
            
        Returns
        -------
        pd.DataFrame
            Cleaned NIST data
        """
        df = pd.read_csv(file)
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace('="', '').str.replace('"', '').str.strip()
        df['obs_wl_vac(nm)'] = pd.to_numeric(df['obs_wl_vac(nm)'], errors='coerce') / 1000
        df['intens'] = pd.to_numeric(df['intens'], errors='coerce')
        df = df.dropna(subset=['obs_wl_vac(nm)'])
        return df
    
    @staticmethod
    def build_line_dict(nist_df, elements=['Na', 'Mg', 'Fe', 'Ti'], 
                       min_intensity=0, top_n=None):
        """
        Build dictionary of spectral lines for specified elements.
        
        Parameters
        ----------
        nist_df : pd.DataFrame
            NIST data from load_nist_data()
        elements : list
            List of element symbols
        min_intensity : float
            Minimum line intensity threshold
        top_n : int, optional
            Keep only top N lines per element
            
        Returns
        -------
        dict
            Dictionary mapping element to wavelength arrays
        """
        line_dict = {}
        for elem in elements:
            mask = (nist_df['element'] == elem) & (nist_df['intens'] >= min_intensity)
            subset = nist_df[mask].copy()

            if top_n is not None:
                subset = subset.sort_values(by='intens', ascending=False).head(top_n)

            line_dict[elem] = subset['obs_wl_vac(nm)'].values
        return line_dict
    
    def match_NIST_lines(self, line_dict, tolerance=0.01, prominence=0.2, region=None):
        """
        Match known spectral lines in the spectrum.
        
        Parameters
        ----------
        line_dict : dict
            Dictionary from build_line_dict()
        tolerance : float
            Maximum wavelength difference for a match
        prominence : float
            Peak prominence threshold for detection
        region : tuple, optional
            (lower, upper) wavelength region to search
            
        Returns
        -------
        list
            List of matched lines with metadata
        """
        inverted = 1 - self.flux
        peaks, _ = find_peaks(inverted, height=prominence)

        matched = []
        for idx in peaks:
            wl = self.wavelengths[idx]
            if region and not (region[0] <= wl <= region[1]):
                continue

            closest = None
            min_diff = tolerance
            for elem, lines in line_dict.items():
                for ref in lines:
                    diff = abs(wl - ref)
                    if diff < min_diff:
                        closest = {"element": elem, "wavelength": wl, 
                                 "reference": ref, "index": idx}
                        min_diff = diff
            if closest:
                matched.append(closest)
        
        self.matched_lines = matched
        return matched
    
    def match_ESPRESSO_LINES(self, file_path, tolerance=0.01, prominence=0.2, region=None):
        """ 
        Match known ESPRESSO spectral lines with the EW values 
        
        Parameters
        ----------
        file_path : str 
            Path to ESPRESSO lines file
            
        Returns
        -------
        list
            List of matched lines with metadata
        """
        espresso_df = pd.read_csv(file_path, delim_whitespace=True, header = None, names = ['wavelength', 'EW'])
        # change the wavelengths to microns
        # 
        # convert waveelengths from angstroms to microns
        espresso_df['wavelength'] = espresso_df['wavelength'] * 1e-4
        # convert EW from mA to nm
        inverted = 1 - self.flux
        peaks, _ = find_peaks(inverted, height=prominence)

        matched = []
        for idx in peaks:
            wl = self.wavelengths[idx] * 1e-4 # Convert from angstroms to microns
            if region and not (region[0] <= wl <= region[1]):
                continue

            closest = None
            min_diff = tolerance
            for elem, lines in espresso_df.items():
                for ref in lines:
                    diff = abs(wl - ref)
                    if diff < min_diff:
                        closest = {"element": elem, "wavelength": wl, 
                                 "reference": ref, "index": idx}
                        min_diff = diff
            if closest:
                matched.append(closest)
        
        self.matched_lines = matched
        return matched

    def build_mask(self, matches=None, buffer=1.2):
        """
        Build mask to exclude spectral lines based on FWHM.
        
        Parameters
        ----------
        matches : list or pd.DataFrame, optional
            Line matches (uses self.matched_lines if None)
        buffer : float
            Multiplier to make masked region wider than FWHM
            
        Returns
        -------
        tuple
            (mask array, list of exclude regions)
        """
        if matches is None:
            if self.matched_lines is None:
                raise ValueError("No matched lines available. Run match_lines() first.")
            matches = self.matched_lines
        
        mask = np.ones(len(self.wavelengths), dtype=bool)
        exclude_regions = []

        # Handle matches as either DataFrame or list of dicts
        if isinstance(matches, pd.DataFrame):
            matches_iter = matches.iterrows()
            get_idx = lambda row: row[1]['index'] if 'index' in row[1] else np.argmin(
                np.abs(self.wavelengths - row[1].get('wavelength', row[1].get('observed_wl')))
            )
        else:
            matches_iter = enumerate(matches)
            get_idx = lambda item: item[1]['index']

        for item in matches_iter:
            idx = get_idx(item)
            
            # Estimate FWHM using the peak width in index space
            region_flux = 1 - self.flux
            results_half = peak_widths(region_flux, [idx], rel_height=0.5)
            fwhm_pixels = results_half[0][0]

            dlambda = np.gradient(self.wavelengths)
            fwhm_lambda = fwhm_pixels * dlambda[idx]
            width = buffer * fwhm_lambda
            center = self.wavelengths[idx]
            lower = center - width
            upper = center + width
            exclude_regions.append((lower, upper))

            mask &= (self.wavelengths < lower) | (self.wavelengths > upper)
        
        self.mask = mask
        self.exclude_regions = exclude_regions
        return mask, exclude_regions
    
    def plot(self, region=None, show_matches=False, show_mask=False, figsize=(12, 6)):
        """
        Plot the spectrum with optional line matches and mask.
        
        Parameters
        ----------
        region : tuple, optional
            (lower, upper) wavelength region to plot
        show_matches : bool
            Whether to show matched lines
        show_mask : bool
            Whether to show masked regions
        figsize : tuple
            Figure size
        """
        plt.figure(figsize=figsize)
        plt.plot(self.wavelengths, self.flux, label='Spectrum', color='black')
        
        legend_elements = [Line2D([0], [0], color='black', lw=2, label='Spectrum')]
        
        if show_mask and self.mask is not None:
            plt.plot(self.wavelengths[self.mask], self.flux[self.mask], 
                    label='Masked Spectrum', color='red', alpha=0.7)
            legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Masked'))
        
        if show_matches and self.matched_lines:
            color_map = {
                'Fe': 'orange', 'Na': 'blue', 'Mg': 'green', 'Ti': 'purple',
                'He': 'red', 'Ca': 'brown', 'Si': 'pink'
            }
            used_elements = set()
            
            for match in self.matched_lines:
                if region and not (region[0] <= match['wavelength'] <= region[1]):
                    continue
                    
                element = match['element']
                color = color_map.get(element, 'gray')
                used_elements.add(element)
                plt.axvline(match['wavelength'], color=color, linestyle='--', 
                          alpha=0.7, linewidth=1)
            
            for element in sorted(used_elements):
                color = color_map.get(element, 'gray')
                legend_elements.append(Line2D([0], [0], color=color, lw=2,
                                            linestyle='--', label=f'{element} Lines'))
        
        plt.legend(handles=legend_elements, loc='best')
        
        if region:
            plt.xlim(region)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux')
        title = 'Spectrum'
        if show_matches:
            title += ' with Matched Lines'
        if show_mask:
            title += ' (Masked)'
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_molecfit_config(self, include_regions=None, exclude_regions=None, 
                           precision=7):
        """
        Generate Molecfit configuration strings.
        
        Parameters
        ----------
        include_regions : list of tuple, optional
            Wavelength regions to include (uses full range if None)
        exclude_regions : list of tuple, optional
            Wavelength regions to exclude (uses self.exclude_regions if None)
        precision : int
            Number of decimal places
            
        Returns
        -------
        dict
            Dictionary with 'WAVE_INCLUDE' and 'WAVE_EXCLUDE' keys
        """
        if exclude_regions is None:
            exclude_regions = self.exclude_regions
        
        fmt = f"{{:.{precision}f}}"
        config = {}
        
        if include_regions:
            include_flat = [val for pair in include_regions for val in pair]
            config['WAVE_INCLUDE'] = ",".join(fmt.format(val) for val in include_flat)
        
        if exclude_regions:
            exclude_flat = [val for pair in exclude_regions for val in pair]
            config['WAVE_EXCLUDE'] = ",".join(fmt.format(val) for val in exclude_flat)
        
        return config
    
    def print_molecfit_config(self, include_regions=None, exclude_regions=None, 
                             precision=7):
        """Print Molecfit configuration in .rc format."""
        config = self.get_molecfit_config(include_regions, exclude_regions, precision)
        for key, value in config.items():
            print(f"{key} = {value}")


#%% Example usage
if __name__ == "__main__":
    # Load spectrum
    spec = Spectrum("wav_cal_optimal_red_nspec200714_0226_70.spectrum")
    
    # Define region of interest
    region2 = (1.0812, 1.0847)
    
    # Load NIST data and build line dictionary
    nist_df = Spectrum.load_nist_data('./data/NIST_lines.csv')
    line_dict = Spectrum.build_line_dict(nist_df)
    
    # Match lines
    matched = spec.match_NIST_lines(line_dict, tolerance=0.01, prominence=0.16, region=region2)
    
    # Build mask
    mask, exclude_regions = spec.build_mask(buffer=1.1)
    
    # Plot with matches and mask
    spec.plot(region=region2, show_matches=True, show_mask=True)
    
    # Get Molecfit config
    spec.print_molecfit_config(include_regions=[region2], precision=7)
# %%
 
# Run this with the expresso.txt files to see if it works with those as well.
    espresso_matched = spec.match_ESPRESSO_LINES('./data/K2_espresso.txt', tolerance=0.01, prominence=0.16, region=region2)

# %%
    # Use the espresso matched lines to build a mask
    mask_espresso, exclude_regions_espresso = spec.build_mask(matches=espresso_matched, buffer=1.1)
# %%
    # plot the spectrum with the espresso matched lines and mask
    spec.plot(region=region2, show_matches=True, show_mask=True)
    spec.plot(region=region2)
# %%
