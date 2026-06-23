# WLIIA (Whose Line Is It Anyway)

A Python-based tool for spectral line identification and masking in astronomical spectra. WLIIA helps astronomers identify absorption/emission lines, match them against known catalogs (NIST, ESPRESSO), and generate masks for use with tools like Molecfit.

## Features

- **Spectral Line Matching**: Automatically identify spectral lines by matching against:
  - NIST Atomic Spectra Database
  - ESPRESSO stellar template line lists
  - Custom line catalogs
  
- **Flexible Masking Methods**: 
  - FWHM-based masking
  - Equivalent Width (EW)-based masking
  - Adjustable buffer parameters for mask width
  
- **Interactive Web Interface**: Streamlit-based UI for:
  - Uploading FITS, CSV, or .spectrum files
  - Visualizing spectra with telluric models
  - Adjusting matching parameters (tolerance, prominence)
  - Selecting wavelength regions
  - Generating Molecfit-compatible wavelength exclusion lists

- **Visualization**: Color-coded line identification by element with customizable plots

## Installation

### Requirements
```bash
pip install numpy pandas scipy astropy matplotlib streamlit
```

### Clone the Repository
```bash
git clone https://github.com/chase-urasaki/WLIIA.git
cd WLIIA
```

## Usage

### Web Interface (Recommended)

Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```

This provides an interactive interface where you can:
1. Upload your spectrum file (FITS/CSV/.spectrum format)
2. Optionally upload a telluric model (FITS)
3. Select a wavelength region to analyze
4. Adjust matching parameters (prominence, tolerance)
5. Choose line catalog (NIST default, or upload ESPRESSO)
6. Generate masks and export Molecfit wavelength exclusion lists

### Python API

```python
from WLIIA import Spectrum, load_spectrum, clean_nist_csv, build_line_dict

# Load your spectrum
spectrum = load_spectrum("your_spectrum.fits")

# Load and prepare NIST line catalog
nist_df = clean_nist_csv("NIST_lines.csv")
line_dict = build_line_dict(nist_df, elements=['Fe', 'Na', 'Mg'], top_n=50)

# Match lines in a specific wavelength region
region = (0.650, 0.660)  # in microns
matches = spectrum.match_known_lines(
    line_dict, 
    region=region, 
    tolerance=0.01,      # matching tolerance in microns
    prominence=0.15      # minimum line depth
)

# Generate a mask
mask = spectrum.build_line_mask_FWHM(matches, buffer=1.2)

# Visualize results
spectrum.plot_with_tellurics(matches=matches, region=region, mask=mask)
```

## File Formats

### Supported Input Formats
- **FITS**: Standard astronomical FITS files with wavelength/flux tables
- **CSV**: Comma-separated values with wavelength and flux columns
- **.spectrum**: Whitespace-delimited text files with wavelength and flux

### Output
- Molecfit-compatible wavelength inclusion/exclusion regions
- DataFrame of matched lines with element identification
- Boolean masks for spectral filtering

## Core Functions

- `load_spectrum(file_path)`: Load spectrum from file
- `match_known_lines()`: Match observed lines to catalog
- `match_espresso_lines()`: Match using ESPRESSO templates
- `build_line_mask_FWHM()`: Create mask based on line FWHM
- `build_line_mask_EW()`: Create mask based on equivalent width
- `print_molecfit_rc_lines()`: Generate Molecfit configuration

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See LICENSE file for details.

## Authors

Chase Urasaki

## Acknowledgments

- NIST Atomic Spectra Database
- ESO ESPRESSO pipeline line lists
- Molecfit telluric correction tool
