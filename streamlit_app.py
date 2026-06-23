# wliia_webapp.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from io import StringIO
import importlib
from pathlib import Path
from WLIIA import Spectrum, load_espresso_lines, clean_nist_csv, build_line_dict, print_molecfit_rc_lines, load_spectrum, plot_mask

# Check package dependencies
RecursionError = RecursionError  # to avoid linter warnin
st.set_page_config(page_title="Whose Line Is It Anyway?", layout="wide")
st.title("Whose Line Is It Anyway — Spectral Line Masking Tool")

# Subtitle 
st.subheader("A tool for identifying and masking spectral lines in astronomical spectra.")

# # Sidebar controls
# st.sidebar.header("Options")
# st.sidebar.write("Adjust parameters and upload files below.")
# line_source = st.sidebar.selectbox("Line source", ["NIST", "ESPRESSO"])
# region = st.sidebar.text_input("Region (microns)", "0.652,0.654")
# prominence = st.sidebar.slider("Prominence", 0.01, 0.5, 0.1)
# tolerance = st.sidebar.slider("Wavelength Tolerance (micron)", 0.001, 0.05, 0.01)
# mask_method = st.sidebar.radio("Masking method", ["FWHM", "EW"])

# Uploads
spectrum_file = st.file_uploader("Upload spectrum (FITS)", type=["fits", "csv"])
telluric_file = st.file_uploader("Upload telluric model (optional, FITS)", type=["fits"])

# --------------------------
# Session defaults
# --------------------------
if "spectrum" not in st.session_state:
    st.session_state.spectrum = None
if "telluric" not in st.session_state:
    st.session_state.telluric = None
if "show_tell" not in st.session_state:
    st.session_state.show_tell = True
if "tell_alpha" not in st.session_state:
    st.session_state.tell_alpha = 0.5
if "tell_scale" not in st.session_state:
    st.session_state.tell_scale = 1.0
if "line_source" not in st.session_state:
    st.session_state.line_source = "NIST"
if "matches" not in st.session_state:
    st.session_state.matches = pd.DataFrame()  # Empty DataFrame instead of None
if "line_dict" not in st.session_state:
    st.session_state.line_dict = {}
if "generate_mask" not in st.session_state:
    st.session_state.generate_mask = False


# --------------------------
# Uploader & loaders
def load_spectrum_file(file):
    """
    Load a spectrum from a file upload and return a Spectrum instance.
    
    Parameters
    ----------
    file : UploadedFile
        A Streamlit uploaded file object
        
    Returns
    -------
    Spectrum
        A Spectrum object containing the loaded data
    """
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        with fits.open(file, memmap=False) as hdul:
            # try table HDU
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None and hasattr(hdu.data, "columns"):
                    arr = np.array(hdu.data).byteswap().newbyteorder()
                    df = pd.DataFrame(arr)
                    break
            else:
                # image: try first image as 1D
                data = np.squeeze(np.array(hdul[0].data))
                x = np.arange(data.size)
                df = pd.DataFrame({"wavelength": x, "flux": data})

    # guess columns
    cols = {c.lower(): c for c in df.columns}
    w = cols.get("wavelength", list(df.columns)[0])
    f = cols.get("flux", list(df.columns)[1] if len(df.columns) > 1 else list(df.columns)[0])
    
    # Create DataFrame with standardized column names
    spectrum_df = df[[w,f]].rename(columns={w:"wavelength", f:"flux"}).dropna().astype(float)
    
    # Create a Spectrum instance
    spectrum_data = {
        "wavelength": spectrum_df["wavelength"].values,
        "flux": spectrum_df["flux"].values
    }
    
    return Spectrum(spectrum_data)

def load_telluric(file):
    with fits.open(file, memmap=False) as hdul:
        # table with wave+val
        for h in hdul:
            d = getattr(h, "data", None)
            if d is None: continue
            if hasattr(d, "columns"):
                cols = {c.name.lower(): c.name for c in d.columns}
                wave_key = next((cols[k] for k in ["wavelength","wave","wl","lambda","micron","um","angstrom","nm"] if k in cols), None)
                val_key  = next((cols[k] for k in ["transmission","telluric","tau","model","flux"] if k in cols), None)
                if wave_key and val_key:
                    return {"wave": np.asarray(d[wave_key]).astype(float),
                            "val":  np.asarray(d[val_key]).astype(float)}
            # 1D vector
            arr = np.squeeze(np.asarray(d))
            if arr.ndim == 1:
                return {"wave": None, "val": arr.astype(float)}
        # fallback to primary image
        arr = np.squeeze(np.asarray(hdul[0].data))
        if arr.ndim == 1:
            return {"wave": None, "val": arr.astype(float)}
    return None

# Stash uploads into session state
if spectrum_file:
    try:
        # Load as Spectrum object
        spectrum_obj = load_spectrum_file(spectrum_file)
        
        # Store both the object and the data in session state
        st.session_state.spectrum_obj = spectrum_obj  # Store the actual Spectrum object
        st.session_state.spectrum = {  # Store data as dict for compatibility with existing code
            "wavelength": pd.Series(spectrum_obj.data["wavelength"]),
            "flux": pd.Series(spectrum_obj.data["flux"])
        }
        st.toast(f"Loaded spectrum: {len(spectrum_obj.data['wavelength'])} points", icon="✅")
    except Exception as e:
        st.error(f"Failed to load spectrum: {e}")

if telluric_file:
    try:
        st.session_state.telluric = load_telluric(telluric_file)
        st.toast("Loaded telluric model", icon="🌫️")
    except Exception as e:
        st.error(f"Failed to load telluric: {e}")

# --------------------------
# LIVE PREVIEW (always visible)
# --------------------------
plot_area = st.empty()

def update_plot():
    fig, ax = plt.subplots(figsize=(8, 4))  # Reduce the figure height
    
    if st.session_state.spectrum is not None:
        w = st.session_state.spectrum.data["wavelength"]
        f = st.session_state.spectrum.data["flux"]
        ax.plot(w, f, label="Spectrum")
        
        # telluric overlay if present & enabled
        if st.session_state.telluric and st.session_state.show_tell:
            tw, tv = st.session_state.telluric["wave"], st.session_state.telluric["val"]
            if tw is not None:
                # interpolate to spectrum grid
                good = np.isfinite(tw) & np.isfinite(tv)
                order = np.argsort(tw[good])
                t_on_spec = np.interp(w, tw[good][order], tv[good][order], left=np.nan, right=np.nan)
            else:
                if len(tv) == len(w):
                    t_on_spec = tv
                else:
                    t_on_spec = None
                    st.warning("Telluric length mismatch and no wavelength axis to interpolate.")
            if t_on_spec is not None:
                ax.plot(w, st.session_state.tell_scale * t_on_spec, alpha=st.session_state.tell_alpha, label="Telluric")
        
        # Overlay matched lines with different colors based on element
        if isinstance(st.session_state.matches, pd.DataFrame):
            # Define color mapping for elements
            color_map = {
                'Fe': 'orange', 
                'Na': 'blue', 
                'Mg': 'green', 
                'Ti': 'purple', 
                'He': 'red'
            }
            default_color = 'gray'  # For unknown elements
            
            # Track which elements are used for the legend
            used_elements = set()
            
            # Plot NIST lines with element information
            if "element" in st.session_state.matches.columns:
                for _, match in st.session_state.matches.iterrows():
                    element = match.get('element', 'Unknown')
                    color = color_map.get(element, default_color)
                    used_elements.add(element)
                    ax.axvline(match['wavelength'], color=color, linestyle='--', 
                               linewidth=0.8, alpha=0.6)
            
            # Plot ESPRESSO lines (no element info)
            elif "catalog_wl" in st.session_state.matches.columns:
                espresso_color = 'darkgreen'
                used_elements.add('ESPRESSO')
                for _, match in st.session_state.matches.iterrows():
                    ax.axvline(match['observed_wl'], color=espresso_color, 
                               linestyle='-.', linewidth=0.8, alpha=0.6)
                # Add a representative line for the legend
                ax.axvline(-999, color=espresso_color, linestyle='-.', 
                           linewidth=0.8, alpha=0.6, label='ESPRESSO Lines')
            
            # Add legend entries for elements
            for element in sorted(used_elements):
                if element != 'ESPRESSO' and element in color_map:
                    ax.axvline(-999, color=color_map[element], linestyle='--',
                               linewidth=0.8, alpha=0.6, label=f'{element} Lines')
            
        # Set the x-limits if region is defined
        if "region_min" in st.session_state and "region_max" in st.session_state:
            ax.set_xlim(st.session_state.region_min, st.session_state.region_max)
    else:
        ax.text(0.5, 0.5, "Upload a spectrum to see the preview", 
                ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Flux")
    
    # Only add legend if we have labels
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
        
    # Update the plot area with our new figure
    plot_area.pyplot(fig)

# Call the update function once to initialize the plot
update_plot()
# plot_area = st.container()
# with plot_area:
#     fig, ax = plt.subplots()
#     if st.session_state.spectrum is not None:
#         w = st.session_state.spectrum["wavelength"].to_numpy()
#         f = st.session_state.spectrum["flux"].to_numpy()
#         ax.plot(w, f, label="Spectrum")
#         # telluric overlay if present & enabled
#         if st.session_state.telluric and st.session_state.show_tell:
#             tw, tv = st.session_state.telluric["wave"], st.session_state.telluric["val"]
#             if tw is not None:
#                 # interpolate to spectrum grid
#                 good = np.isfinite(tw) & np.isfinite(tv)
#                 order = np.argsort(tw[good])
#                 t_on_spec = np.interp(w, tw[good][order], tv[good][order], left=np.nan, right=np.nan)
#             else:
#                 if len(tv) == len(w):
#                     t_on_spec = tv
#                 else:
#                     t_on_spec = None
#                     st.warning("Telluric length mismatch and no wavelength axis to interpolate.")
#             if t_on_spec is not None:
#                 ax.plot(w, st.session_state.tell_scale * t_on_spec, alpha=st.session_state.tell_alpha, label="Telluric")
#         # overlay matched lines (if you’ve run matching)
#         if not st.session_state.matches.empty and {"wavelength"}.issubset(st.session_state.matches.columns):
#             for x in st.session_state.matches["wavelength"].to_numpy():
#                 ax.axvline(x, linestyle="--", linewidth=0.8, alpha=0.5)
#     else:
#         st.info("Upload a spectrum to see the live preview.")
#     ax.set_xlabel("Wavelength (μm)")
#     ax.set_ylabel("Flux")
#     ax.legend(loc="best")
#     st.pyplot(fig, use_container_width=True, clear_figure=True)

# --------------------------
# CONTROLS (change → rerun → preview updates)
# --------------------------
with st.expander("Telluric overlay", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        show_tell_checkbox = st.checkbox("Show telluric", key="show_tell")
    with col2:
        st.slider("Alpha", 0.0, 1.0, step=0.05, key="tell_alpha")
    with col3:
        st.number_input("Scale ×", 0.0, 10.0, step=0.1, key="tell_scale")
st.subheader("Line catalog & matching")
st.session_state.line_source = st.radio("Catalog source", ["NIST","ESPRESSO","UPLOAD"], index=(0 if st.session_state.line_source=="NIST" else 1), horizontal=True)

if st.session_state.line_source == "NIST":
    NIST_PATH = Path(__file__).parent / "NIST_lines.csv"

    @st.cache_data
    def load_nist_default():
        return clean_nist_csv(NIST_PATH)


    nist_df = load_nist_default()
    st.success(f"Loaded default NIST line list: {nist_df.shape[0]} lines", icon="📚")

    # Build the line dictionary 
    line_dictionary = build_line_dict(nist_df) # all elements, no top_n limit


    # nist_file = st.file_uploader("NIST line list (CSV)", type=["csv"], key="nist_lines")
    # if nist_file and ss.spectrum is not None and region:
    #     elements = st.multiselect("Elements", ["He","Fe","Na","Mg","Ti","Si","Ca","K","Ni","Cr"], default=["He","Fe"])
    #     top_n = st.number_input("Top N/element", 1, 200, 20, 1)
    #     if st.button("Run matching (NIST)"):
    #         nist_df = clean_nist_csv(nist_file)
    #         line_dict = build_line_dict(nist_df, elements=elements, top_n=top_n)
    #         ss.matches = pd.DataFrame(
    #             Spectrum.match_known_lines(line_dict, tolerance=tolerance, prominence=prominence, region=region)
    #         )
# elif st.session_state.line_source == "ESPRESSO":
#     espresso_file = st.file_uploader("ESPRESSO line list (txt)", type=["txt"], key="espresso_lines")
#     template = st.selectbox("Template", ["Auto","F","G","K","M"], index=0)
#     if espresso_file and st.session_state.spectrum is not None and region:
#         if st.button("Run matching (ESPRESSO)"):
#             espresso_df = load_espresso_lines(espresso_file)
#             # Use the spectrum object
#             if "spectrum_obj" in st.session_state:
#                 region_matches = st.session_state.spectrum_obj.match_espresso_lines(
#                     espresso_df,
#                     region=region
#                 )
#                 st.session_state.matches = pd.DataFrame(region_matches)
#                 update_plot()  # Update the plot with new matches

# Optional: show matches table under controls
# With this safer check:
if isinstance(st.session_state.matches, pd.DataFrame):
    st.subheader("Matched Lines")
    st.dataframe(st.session_state.matches, use_container_width=True)

# Select region over which to match lines, the whole spectrum by default 
# Add this after loading the spectrum
if st.session_state.spectrum is not None:
    w_min = float(st.session_state.spectrum["wavelength"].min())
    w_max = float(st.session_state.spectrum["wavelength"].max())
    
    # Store in session state if not already there
    if "region_min" not in st.session_state:
        st.session_state.region_min = w_min
    if "region_max" not in st.session_state:
        st.session_state.region_max = w_max
    
    # Create a container for the wavelength inputs
    st.subheader("Select Wavelength Region")
    col1, col2 = st.columns(2)
    with col1:
        region_min = st.number_input(
            "Min wavelength (μm)", 
            value=st.session_state.region_min,
            min_value=w_min,
            max_value=w_max-0.001,
            step=0.001,
            format="%.6f",
            key="region_min"
        )
    with col2:
        region_max = st.number_input(
            "Max wavelength (μm)", 
            value=st.session_state.region_max,
            min_value=w_min+0.001,
            max_value=w_max,
            step=0.001,
            format="%.6f",
            key="region_max"
        )
    
    # Update the region tuple
    region = (region_min, region_max)
    
    if st.button("Update Plot Region"):
        update_plot()
    # Add a button to reset to full range

# With this:
if st.button("Reset to Full Range"):
    # Use proper reference to session_state object
    if "spectrum_obj" in st.session_state:
        # Get min and max directly from the stored spectrum data
        w_min = float(st.session_state.spectrum["wavelength"].min())
        w_max = float(st.session_state.spectrum["wavelength"].max())
        
        # Update session state values with a different key to avoid conflicts
        st.session_state["region_min"] = w_min
        st.session_state["region_max"] = w_max
        
        # Force update the plot
        update_plot()
        
        # Rerun the app to refresh the UI components
        st.rerun()  # st.experimental_rerun() is deprecated, use st.rerun() instead

st.subheader("Matching Parameters")
col1, col2 = st.columns(2)

with col1:
    prominence = st.slider(
        "Line Prominence", 
        min_value=0.01, 
        max_value=0.5, 
        value=0.15,  # Default value
        step=0.01,
        help="Higher values detect only deeper absorption lines"
    )
    
with col2:
    tolerance = st.slider(
        "Wavelength Tolerance (μm)", 
        min_value=0.0001, 
        max_value=0.05, 
        value=0.01,  # Default value
        step=0.0001,
        format="%.4f",
        help="Max distance between observed and catalog lines"
    )

# Add a button to run matching with the current parameters
if st.button("Run Line Matching"):
    if st.session_state.spectrum is not None and region and "spectrum" in st.session_state:
        if st.session_state.line_source == "NIST":
            region_matches = st.session_state.spectrum_obj.match_known_lines(
                line_dict=line_dictionary, 
                region=region, 
                tolerance=tolerance,  # Using slider value
                prominence=prominence  # Using slider value
            )
        elif st.session_state.line_source == "ESPRESSO" and "espresso_df" in st.session_state:
            region_matches = st.session_state.spectrum_obj.match_espresso_lines(
                st.session_state.espresso_df,
                region=region,
                tolerance=tolerance,  # Using slider value
                prominence=prominence  # Using slider value
            )
        else:
            st.warning("Please upload an ESPRESSO line list first.")
            region_matches = []
            
# After storing matches in session state (around line 452)
#st.session_state.matches = pd.DataFrame(region_matches)

# Create a dedicated debugging expander
with st.expander("Debug Information", expanded=True):
    st.write("### DataFrame Structure")
    
    # Show shape information
    st.write(f"**Number of matched lines:** {st.session_state.matches.shape[0]}")
    st.write(f"**Number of columns:** {st.session_state.matches.shape[1]}")
    
    # Display column names and types
    if not st.session_state.matches.empty:
        col_info = pd.DataFrame({
            "Column Name": st.session_state.matches.columns,
            "Data Type": [str(st.session_state.matches[col].dtype) for col in st.session_state.matches.columns],
            "Sample Value": [str(st.session_state.matches[col].iloc[0]) if len(st.session_state.matches) > 0 else "N/A" 
                             for col in st.session_state.matches.columns]
        })
        st.write("**Column Information:**")
        st.dataframe(col_info, use_container_width=True)
    else:
        st.info("No matches found - DataFrame is empty")
    
    # Show first few rows
    if not st.session_state.matches.empty:
        st.write("**First 5 Rows:**")
        st.dataframe(st.session_state.matches.head(), use_container_width=True)
    


# After running matching perform masking 
st.subheader("Masking Options")
mask_method = st.radio("Masking method", ["FWHM", "EW"], index=0, horizontal=True)
if st.button("Generate Mask & Plot"):  
    if st.session_state.spectrum is not None and st.session_state.matches is not None:
        if mask_method == "FWHM":
            mask = spectrum_obj.build_line_mask_FWHM(st.session_state.matches, buffer=1.1)

            plot_mask()
        
        else:
            print("Sorry not implemented yet :(")
            # mask = spectrum_obj.build_line_mask_EW(st.session_state.matches)
        
        # Plot with mask
        spectrum_obj.plot_with_tellurics(
            matches=st.session_state.matches,
            telluric=st.session_state.telluric,
            region=region,
            mask=mask
        )

# Make a new plot of the masked region 




# if spectrum_file:
#     st.subheader("Uploaded Spectrum Preview")

#     spectrum = load_spectrum(spectrum_file)

#     # Create a figure
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(spectrum.data['wavelength'], spectrum.data['flux'], label='Spectrum')
    
#     # Handle telluric file if uploaded
#     if telluric_file:
#         # Check if telluric data length matches spectrum length
#         try:
#             with fits.open(telluric_file) as hdul:
#                 telluric = hdul[1].data
#                 if len(spectrum.data['wavelength']) != len(telluric):
#                     st.error("Telluric model length does not match spectrum length.\n" 
#                              "Please upload a telluric model with the same number of data points as the spectrum.\n"
#                              "Continuing without telluric model.")
#                 else:
#                     # Overplot the telluric model on the same figure
#                     ax.plot(spectrum.data['wavelength'], telluric, label='Telluric Model', alpha=0.7)
#                     ax.legend()
#         except Exception as e:
#             st.error(f"Error loading telluric file: {str(e)}")
    
#     # Set labels and title
#     ax.set_xlabel('Wavelength (micron)')
#     ax.set_ylabel('Flux')
#     ax.set_title('Spectrum Preview')
#     ax.legend()
    
#     # Display the plot
#     st.pyplot(fig)

# Add toggle to show/hide tellurics on plot 


# if spectrum_file:
#     with fits.open(spectrum_file) as hdul:
#         data = hdul[1].data
#         wl = np.array(data['wavelength'])
#         flux = np.array(data['flux'])
#         spec = Spectrum({"wavelength": wl, "flux": flux})

#     matches = []
#     exclude_regions = []

#     # Line matching
#     if line_source == "NIST":
#         nist_file = st.file_uploader("Upload NIST line list (CSV)", type=["csv"])
#         if nist_file:
#             nist_df = clean_nist_csv(nist_file)
#             elements = st.sidebar.multiselect("Select elements", ["Fe", "Na", "Mg", "Ti", "He"], default=["Fe"])
#             top_n = st.sidebar.number_input("Top N lines per element (optional)", min_value=1, value=20)
#             line_dict = build_line_dict(nist_df, elements=elements, top_n=top_n)
#             region_tuple = tuple(map(float, region.split(",")))
#             matches = spec.match_known_lines(line_dict, tolerance=tolerance, prominence=prominence, region=region_tuple)

#     elif line_source == "ESPRESSO":
#         espresso_file = st.file_uploader("Upload ESPRESSO line list (txt)", type=["txt"])
#         if espresso_file:
#             espresso_df = load_espresso_lines(espresso_file)
#             region_tuple = tuple(map(float, region.split(",")))
#             matches = spec.match_espresso_lines(espresso_df, tolerance=tolerance, prominence=prominence, region=region_tuple)

#     # Masking
#     if matches:
#         mask = spec.build_line_mask_FWHM(matches) if mask_method == "FWHM" else spec.build_line_mask_EW(matches)
#     else:
#         mask = None

#     # Telluric
#     if telluric_file:
#         telluric = fits.open(telluric_file)[1].data
#         telluric_dict = {"transmission": telluric}
#     else:
#         telluric_dict = None

#     # Plot and show results
#     spec.plot_with_tellurics(matches=matches, telluric=telluric_dict, region=region_tuple, mask=mask)

#     # Print Molecfit WAVE_EXCLUDE
#     # if hasattr(spec, 'exclude_regions'):
#     #     st.subheader("Molecfit WAVE_EXCLUDE")
#     #     with StringIO() as buffer:
#     #         print_molecfit_rc_lines(include_regions=[region_tuple], exclude_regions=spec.exclude_regions)
#     #         st.code(print_molecfit_rc_lines(include_regions=[region_tuple], exclude_regions=spec.exclude_regions))
