# wliia_webapp.py
import streamlit as st
import pandas as pd
import numpy as np
from astropy.io import fits
from io import StringIO
#from WLIIA import Spectrum, load_espresso_lines, clean_nist_csv, build_line_dict, print_molecfit_rc_lines


st.set_page_config(page_title="Whose Line Is It Anyway", layout="wide")
st.title("Whose Line Is It Anyway — Spectral Line Masking Tool")

# Sidebar controls
st.sidebar.header("Options")
line_source = st.sidebar.selectbox("Line source", ["NIST", "ESPRESSO"])
region = st.sidebar.text_input("Region (microns)", "0.652,0.654")
prominence = st.sidebar.slider("Prominence", 0.01, 0.5, 0.1)
tolerance = st.sidebar.slider("Wavelength Tolerance (micron)", 0.001, 0.05, 0.01)
mask_method = st.sidebar.radio("Masking method", ["FWHM", "EW"])

# Uploads
spectrum_file = st.file_uploader("Upload spectrum (FITS)", type=["fits"])
telluric_file = st.file_uploader("Upload telluric model (optional, FITS)", type=["fits"])

if spectrum_file:
    with fits.open(spectrum_file) as hdul:
        data = hdul[1].data
        wl = np.array(data['wavelength'])
        flux = np.array(data['flux'])
        spec = Spectrum({"wavelength": wl, "flux": flux})

    matches = []
    exclude_regions = []

    # Line matching
    if line_source == "NIST":
        nist_file = st.file_uploader("Upload NIST line list (CSV)", type=["csv"])
        if nist_file:
            nist_df = clean_nist_csv(nist_file)
            elements = st.sidebar.multiselect("Select elements", ["Fe", "Na", "Mg", "Ti", "He"], default=["Fe"])
            top_n = st.sidebar.number_input("Top N lines per element (optional)", min_value=1, value=20)
            line_dict = build_line_dict(nist_df, elements=elements, top_n=top_n)
            region_tuple = tuple(map(float, region.split(",")))
            matches = spec.match_known_lines(line_dict, tolerance=tolerance, prominence=prominence, region=region_tuple)

    elif line_source == "ESPRESSO":
        espresso_file = st.file_uploader("Upload ESPRESSO line list (txt)", type=["txt"])
        if espresso_file:
            espresso_df = load_espresso_lines(espresso_file)
            region_tuple = tuple(map(float, region.split(",")))
            matches = spec.match_espresso_lines(espresso_df, tolerance=tolerance, prominence=prominence, region=region_tuple)

    # Masking
    if matches:
        mask = spec.build_line_mask_FWHM(matches) if mask_method == "FWHM" else spec.build_line_mask_EW(matches)
    else:
        mask = None

    # Telluric
    if telluric_file:
        telluric = fits.open(telluric_file)[1].data
        telluric_dict = {"transmission": telluric}
    else:
        telluric_dict = None

    # Plot and show results
    spec.plot_with_tellurics(matches=matches, telluric=telluric_dict, region=region_tuple, mask=mask)

    # Print Molecfit WAVE_EXCLUDE
    # if hasattr(spec, 'exclude_regions'):
    #     st.subheader("Molecfit WAVE_EXCLUDE")
    #     with StringIO() as buffer:
    #         print_molecfit_rc_lines(include_regions=[region_tuple], exclude_regions=spec.exclude_regions)
    #         st.code(print_molecfit_rc_lines(include_regions=[region_tuple], exclude_regions=spec.exclude_regions))
