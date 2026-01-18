"""
    Whoppsel : The echelle-spectrograph analysis module
    ===================================================

    A toolkit for automated echelle-spectrograph data analysis.

    Provides functions for order detection, tracing, extraction, cross-correlation
    analysis, and V-curve fitting for focuser position optimization. Designed as a
    flexible module - use functions independently or combine them for custom
    workflows.

    Functions
    ----------
        order_detector      - Detect echelle-orders in 2D images
        order_tracer        - Trace order positions across detector
        order_extractor     - Extract 1D spectra from 2D orders
        cross_correlate     - Measure spectral line broadening
        v_curve_fitting     - Determine optimal focuser position
        simple_plot         - Quick line plotting utility
    
    Constants
    ----------
        DEFAULT_FIG_WIDTH   - Default figure width (6.32inches, LaTeX textwidth)
        DEFAULT_FIG_HEIGHT  - Default figure height (3.91 inches, golden ratio)

    Basic usage example
    ----------
    >>> import whoppsel as wpl
    >>> 
    >>> # Load CCD images, image data is stored in separate arrays named imagedata_flat and imagedata_calib
    >>> imagedata_flat  = fits.getdata('echelle_image_flat.fits')
    >>> imagedata_calib = fits.getdata('echelle_image_calibration.fits')
    >>>
    >>> # Detect orders from flat image
    >>> detected_orders = wpl.order_detector(imagedata_flat)
    >>>
    >>> # Trace orders from flat image, using detected_orders
    >>> traced_orders = wpl.order_tracer(imagedata_flat, detected_orders)
    >>>
    >>> # Extract orders from calibration image, using traced_orders
    >>> extracted_orders = wpl.order_extractor(imagedata_calib, traced_orders)
    >>>
    >>> # Perform cross-correlation analysis over multiple different exposures
    >>> datasets    = [ext_ords_1, ext_ords_2, ext_ords_3]  # Extracted from datasets gathered at different focuser positions
    >>> positions   = [40000, 50000, 60000] # Focuser position 
    >>>
    >>> result = wpl.cross_correlate(
    >>>     datasets,
    >>>     positions,
    >>>     dataset_base_corr   = 0,
    >>>     ord_to_corr         = 52
    >>> )
    >>>
    >>> # Fit V-curve to find optimal focuser position
    >>> optimal_focus, errors = wpl.v_curve_fitting(
    >>>     positions,      
    >>>     result[1],
    >>>     result[2],
    >>>     slope_start_distance = 2,
    >>>     current_order = 52
    >>> )
    >>>
    >>> print(f"Optimal focus: {optimal_focus[2]:.1f} ± {errors[2]:.1f}")

    Notes
    ----------
    In it's current state the toolkit expects the user to have pre-processed spectrographic images,
    the transformation functions associated with raw data transformation have not been implemented yet.
"""


# ====================== IMPORTS ======================
# Standard libraries
import warnings
import os

# Scientific computing
import numpy    as  np
import pandas   as  pd

# Scipy
from scipy.signal   import find_peaks, peak_widths, correlate, correlation_lags
from scipy.optimize import curve_fit, least_squares

# Plotting/graphing
import matplotlib.pyplot as plt

# ====================== METADATA ======================
__all__ = [
    # Main functions
    'order_detector',
    'order_tracer',
    'order_extractor',
    'cross_correlate',
    'v_curve_fitting',

    # Public utility
    'simple_plot',

    # Public constants
    'DEFAULT_FIG_WIDTH',
    'DEFAULT_FIG_HEIGHT',
]
__version__ = '0.2.0'
__author__  = 'Karl Nõupuu'
__email__   = 'karl.noupuu@hotmail.com'


# ====================== GLOBAL PREAMBLE ====================== 
DEFAULT_FIG_WIDTH = 455.22408 * (1 / 72)
DEFAULT_FIG_HEIGHT = DEFAULT_FIG_WIDTH / ((1 + 5 ** 0.5) / 2)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_STYLE_FILE = os.path.join(_SCRIPT_DIR, 'whoppsel.mplstyle')

plt.style.use(_STYLE_FILE)


# ====================== MAIN FUNCTIONS ======================
def order_detector(
        image_data          : np.ndarray,
        vert_slice_idx      : int   = 1024,
        vert_slice_radius   : int   = 10,
        max_signal          : int   = 6000000,
        oversampling_coeff  : int   = 100,
        gauss_width_at      : float = 0.5,
        ref_order           : int   = 52,
        ref_order_pix       : int   = 1220
    ) -> pd.DataFrame:
    '''
    Detects echelle orders in a 2D spectrographic image.

    Uses vertical slice analysis and Gaussian fitting to identify
    spectral order locations and widths. Creates a 1D spectrum
    by summing vertical slices centered at vert_slice_idx ±
    vert_slice_radius, then performs peak detection and refinement
    via Gaussian fitting.

    Algorithm:
        1. Initial peak detection using scipy.signal.find_peaks
        2. Gaussian fitting at each peak for increased accuraccy
        3. Calculate FWHM and edge locations for each detected order
        4. Assign order numbers relative to reference order 52

    Parameters
    ----------
    image_data : ndarray 
        2D CCD-image data (y-axis x x-axis pixels)
    vert_slice_idx : int, default = 1024
        X-axis pixel location for the central vertical slice
    vert_slice_radius : int, default = 10
        Radius (pixels) used for vertical slice summation range
    max_signal : int, default = 6000000
        Maximum signal threshold for valid order peaks (excludes saturated orders)
    oversampling_coeff : int, default = 100
        Coefficient by which the original pixel range is divided for better Gaussian fitting
    gauss_width_at : float, default = 0.5
        Relative height for (0 - 1.0) for Gaussian width measurement (0.5 = FWHM)
    ref_order : int, default = 52
        Known order number of the reference order (usually an order that is always present and detectable)
    ref_order_pix : int, default = 1220
        Estimated Y-pixel position of the reference order

    Returns
    ----------
    detected_orders : pd.DataFrame
        DataFrame containing detected order parameters:
        - order_nums        : Order numbers (relative to order 52)
        - order_centers     : Peak center positions (pixels)
        - order_low_edge    : Lower edge of each order (pixels)
        - order_high_edge   : Upper edge of each order (pixels)
        - order_fwhms       : Full width at half maximum values
    '''
    
    # ----- Input validation -----
    nx, ny = np.shape(image_data)

    # IndexError for out of bounds indexing
    if (vert_slice_idx > (nx - 1)) or (vert_slice_idx < 0):
        raise IndexError(f"x-axis index {vert_slice_idx} out of bounds for image shape [{0}:{nx - 1}, {0}:{ny - 1}]")
    
    # ValueError for gaussian width
    if (gauss_width_at <= 0) or (gauss_width_at > 1.0):
        raise ValueError(f"Gaussian width percentage should fall into the range (0.0, 1.0]")
    
    # ValueError for order_pix52_loc
    if (ref_order_pix > (nx - 1)) or (ref_order_pix < 0):
        raise ValueError(f"y-axis index for reference order location {ref_order_pix} out of bounds for image shape [{0}:{nx - 1}, {0}:{ny - 1}]")

    # Warning for vertical slice radius, that might result in a non uniform vertical slice of data
    if ((vert_slice_idx - vert_slice_radius) < 0) or ((vert_slice_idx + vert_slice_radius) > nx):
        warnings.warn(f"Defined vertical slice radius {vert_slice_radius} extends over"
                      f"image bounds (0:{nx}, 0:{ny}).", UserWarning)


    # ----- Preamble -----
    # Defining a slice of the image used for order location detection
    vert_slice = image_data[:,
                            max(0, vert_slice_idx - vert_slice_radius):
                            min(nx, vert_slice_idx + vert_slice_radius)]
    
    # Helper data and summed slice data
    x_data = np.arange(0, 2048, 1)
    y_data = sum(np.transpose(vert_slice))

    # Initial peak detection
    initial_peaks, _ = find_peaks(y_data, prominence = 10. * np.median(y_data))
    initial_peaks = [i for i in initial_peaks if y_data[i] < max_signal]

    # Refined peak detection, based on initial peak data
    peak_fwhms, peak_lows, peak_centers, peak_highs = [], [], [], []


    # ----- Main functionality -----
    for peak in initial_peaks:
        peak_lo = peak - int(np.round(np.mean(np.diff(initial_peaks))) / 2)
        peak_hi = peak + int(np.round(np.mean(np.diff(initial_peaks))) / 2)
        peak_x_data = x_data[peak_lo:peak_hi]
        peak_y_data = y_data[peak_lo:peak_hi]

        # Oversampling_coeff used for sub-pixling, used for refined peak detection
        # using Gaussian fit results
        peak_x_fit_data = np.arange(min(peak_x_data), max(peak_x_data), 1 / oversampling_coeff)

        # Gaussian fitting
        try:
            popt, _ = curve_fit(_gaussian, peak_x_data, peak_y_data, p0 = [y_data[peak], x_data[peak], 0.1])
        except:
            continue

        gaussian_fit    = _gaussian(peak_x_fit_data, *popt)
        refit_peaks, _  = find_peaks(gaussian_fit, prominence = 1)
        refit_fwhms     = peak_widths(gaussian_fit, refit_peaks, gauss_width_at)

        # Adding results to lists, refit values are divided by oversampling_coeff to bring them back
        # to the relevant index range
        peak_fwhms.append(  refit_fwhms[0][0]   / oversampling_coeff)
        peak_lows.append(   int(np.floor(refit_fwhms[2][0] / oversampling_coeff)) + x_data[peak_lo])
        peak_centers.append(int(np.floor(refit_peaks[0]    / oversampling_coeff)) + x_data[peak_lo])
        peak_highs.append(  int(np.floor(refit_fwhms[3][0] / oversampling_coeff)) + x_data[peak_lo])

    detected_orders = pd.DataFrame(data = {
        'order_nums'        : _get_ordernums(np.asarray(peak_centers), ref_order, ref_order_pix) ,
        'order_fwhms'       : peak_fwhms,
        'order_low_edge'    : peak_lows,
        'order_centers'     : peak_centers,
        'order_high_edge'   : peak_highs}
    ).set_index('order_nums')

    return detected_orders


def order_tracer(
        image_data             : np.ndarray,
        order_locs             : pd.DataFrame,
        trace_window_radius    : int = 10,
        trace_poly_order       : int = 2,
        trace_x_min            : int = 800,
        trace_x_max            : int = 1100,
        refine_count           : int = 3,
        refine_poly_order      : int = 4,
        refine_x_min           : int = 500,
        refine_x_max           : int = 1500
    ) -> pd.DataFrame:
    '''
    Traces the echelle-spectra orders using the relevant data on the detected
    orders using order_detector.
    Traces echelle order positions across the CCD detector matrix.

    Uses an iterative box-window approach to map order curvature. Starts
    with a narrow central region (where orders are straighter) and fits a polynomial
    to the detected positions. This polynomial guides subsequent refinement passes
    that extend the trace toward the curved order edges.

    The function stores both initial and refined trace coefficients, allowing comparison
    of trace quality vs. computational cost.

    Algorithm:
        1. Initial trace: Box-window scan in central region (trace_x_min to trace_x_max)
        2. Fit polynomial of degree trace_poly_order to initial positions
        3. Refinement: Use polynomial as guide for extended tracing
        4. Repeat refinement refine_count times with higher-order polynomial. Each repeat increases the maximum extent of the refinement region
        5. Store both initial and refined polynomial coefficients

    Parameters
    ----------
    image_data : ndarray 
        2D CCD-image data
    order_locs : pd.DataFrame
        DataFrame from order_detector() containing order centers and edges
    trace_window_radius : int, default = 10
        Half-height (pixels) of box window in y-axis direction
    trace_poly_order : int, default = 2
        Polynomial degrees for initial trace fitting
    trace_x_min : int, default = 800
        Starting x-axis position (pixels) for initial trace
    trace_x_max : int, default = 1100
        Ending x-axis position (pixels) for initial trace
    refine_count : int, default = 3
        Number of iterative refinement passes
    refine_poly_order : int, default = 4
        Polynomial degree for refined trace fitting
    refine_x_min : int, default = 500
        Maximum starting x-axis position (pixels) for refined trace (each iterative pass extends towards this value)
    refine_x_max : int, default = 1500
        Maximum ending x-axis position (pixels) for refined trace (each iterative pass extends towards this value)

    Returns
    ----------
    order_locs : pd.DataFrame
        Input DataFrame with added columns:
        - init_trace_coeffs: Polynomial coefficients from initial trace
        - ref_trace_coeffs: Polynomial coefficients from refined trace
    '''
    
    nx, ny = np.shape(image_data)
    
    # ----- Input validation -----
    if (trace_x_min < 0) or (trace_x_max > (nx - 1)):
        raise IndexError(f"Trace window index out of bounds for image shape ({nx}, {ny})")
    if (refine_x_min < 0) or (refine_x_max > (nx - 1)):
        raise IndexError(f"Trace refine window index out of bouds for image shape ({nx}, {ny})")

    # ----- Preamble -----
    order_centers = order_locs['order_centers']
    order_poly_coeffs, refined_poly_coeffs = [], []

    x_range = np.arange(trace_x_min, trace_x_max, 1)

    # ----- Main functionality -----
    # Trace algorithm
    for center in order_centers:
        trace = []
        center = int(np.round(center))

        # For all x-axis indices perform Gaussian fitting in box-window y-axis range to determine 
        # accurate order center location at index. This data is then used for polynomial fitting
        for x in x_range:
            
            # Slice arithmetic to ensure no out-of-bounds madness
            hor_slice = slice(max(0, center - trace_window_radius),
                              min(ny, center + trace_window_radius))
            
            try:    
                y_rel = _gaussian_centroid(image_data[hor_slice, x])
            except:
                y_rel = np.array([0])

            if y_rel.size == 0: 
                y_abs = hor_slice.start + 0
            else:               
                y_abs = hor_slice.start + y_rel[0]

            trace.append(y_abs)
        
        # Get trace polynomial coefficients
        trace_coeffs = np.polyfit(x_range, trace, trace_poly_order)
        order_poly_coeffs.append(trace_coeffs)

        # Initial trace iterative refinement
        for ref in range(refine_count):
            refined_trace = []

            # Get guide polynomial from initial trace
            guide_polynomial = np.polyval(trace_coeffs, np.arange(0, nx))
            
            # Iteratively extending box-window region, limited by the maximum allowed extent for the 
            # x-axis range, starts from the initial box-window size, increase by 100px per step.
            # Compared to initial trace, where the Gaussian fitting was done for every pixel, the 
            # Gaussian fitting here is done every 10 pixels.
            xx_range = np.arange(max(0, refine_x_min - ref * 100),
                                 min(nx, refine_x_max + ref * 100), 10)
            
            for xx in xx_range:
                guide_center = int(np.round(guide_polynomial[xx]))

                # Refinement trace window is halved, based on the assumption that the guide trace is 
                # precise enough to contain the order signal in a smaller y-axis range.
                hor_slice = slice(max(0, guide_center - int(trace_window_radius / 2)),
                                  min(ny, guide_center + int(trace_window_radius / 2)))
                
                try:    
                    y_rel = _gaussian_centroid(image_data[hor_slice, xx])
                except: 
                    y_rel = np.array([0])

                if y_rel.size == 0: 
                    y_abs = hor_slice.start + 0
                else:               
                    y_abs = hor_slice.start + y_rel[0]

                refined_trace.append(y_abs)

            trace_coeffs = np.polyfit(xx_range, refined_trace, refine_poly_order)
        
        refined_poly_coeffs.append(trace_coeffs)

    order_locs['init_trace_coeffs'] = order_poly_coeffs
    order_locs['ref_trace_coeffs'] = refined_poly_coeffs

    return order_locs


def order_extractor(
        image_data  : np.ndarray,
        order_locs  : pd.DataFrame,
        ext_x_min   : int = 500,
        ext_x_max   : int = 1500
    ) -> pd.DataFrame:
    '''
    Extracts 1D spectra from 2D echelle orders using traced positions.

    Creates pixel masks for each order based on trace polynomials,
    order boundaries, and order centers, then sums the masked 2D regions along the spatial
    direction to produce 1D spectra. The extraction region is defined
    by the order's lower/upper edges and follows the curved trace across
    the detector.

    Algorithm:
        1. Generate extraction mask using trace coefficients, order edges, and order centers.
        2. For each x-position, identify y-pixels belonging to the order
        3. Sum counts along spatial (y) direction at each wavelength (x) position
        4. Store resulting 1D spectrum [x_data, y_data] for each order

    Parameters
    ----------
    image_data : ndarray 
        2D CCD-image data
    order_locs : pd.DataFrame
        DataFrame from order_tracer() with trace coefficients
    ext_x_min : int, default = 500
        Starting x-axis pixels for extraction region
    ext_x_max : int, default = 1500
        Ending x-axis pixels for extraction region

    Returns
    ---------- 
    order_locs : pd.DataFrame
        Input DataFrame with added column:
        - spectra: List of [x_array, y_array] pairs for each order's 1D spectrum
    '''

    nx, ny = np.shape(image_data)

    # ----- Input validation -----
    if (ext_x_min < 0) or (ext_x_max > (nx - 1)):
        raise IndexError(f"Extraction range index out of bounds for image shape ({nx}, {ny})")

    # ----- Preamble -----
    x_data = np.arange(ext_x_min, ext_x_max, 1)
    spectra = []

    # ----- Main functionality -----
    # Masking and summing for every order in order_locs() dataframe
    for _, order in order_locs.iterrows():

        # Pass to private utility function to get curved mask with shape(n_rows, n_x_pixels)
        y_mask = _spectre_mask(
            x_data,
            order['order_low_edge'],
            order['order_centers'],
            order['order_high_edge'],
            order['ref_trace_coeffs']
        )
        
        # Extract all pixels at once and sum along spatial direction into 1D-spectrum
        spectrum_1d = image_data[y_mask, x_data].sum(axis = 0)
        spectra.append([x_data, spectrum_1d])

    order_locs["spectra"] = spectra
    return order_locs


def cross_correlate(
        datasets                : list,
        analysis_variable       : list,
        dataset_base_corr       : int,
        ord_to_corr             : int,
        correlation_method      : str = 'fft',
        correlation_mode        : str = 'full',
        use_with                : str = 'linear',
        gauss_fit_region        : int = 20,
        show_correlation_plot   : bool = True,
        graph_dump              : bool = False,
        graph_dump_loc          : str = 'graph_dump/correlation'
    ) -> list[np.ndarray]:
    '''
    Measures spectral line broadening across experimental conditions via cross-correlation.
    
    Compares extracted 1D spectra to quantify changes in spectral line width (FWHM).
    Uses cross-correlation to measure pixel-level shifts and Gaussian
    fitting of the correlation peak to achieve sub-pixel precision.

    The correlation peak's FWHM indicates spectral line broadening, which reflects
    focus quality and instrumental stability across different conditions.

    Additionally supports intermediate result graphing when show_correlation_plot = True,
    which plots all the correlation functions for the order being correlated over the entire dataset.

    Algorithm:
        1. Select specified order from all datasets
        2. Cross-correlate each spectrum against the reference spectrum (dataset_base_correlate)
        3. Fit Gaussian (with linear or polynomial background) to correlation peak
        4. Extract peak FWHM and position with uncertainties from covariance matrix
        5. Optionally plot correlation functions for quality control

    Parameters
    ----------
    datasets : list of pd.DataFrame
        List of order_locs DataFrames from order_extract(), one per 
    analysis_variable : list of float
        Independent variable values corresponding to each dataset.
        Used to label measurements when analysing how spectral properties change with experimental conditions.
    dataset_base_corr : int
        Index of reference dataset for cross-correlation
    ord_to_corr : int
        Echelle order number to analyse
    correlation_method : str, default = "fft"
        scipy.signal.correlate method: "fft" (fast) or "direct"
    correlation_mode : str, default = "full"
        scipy.signal.correlate mode: "full", "valid", or "same"
    use_with : str, default = "linear"
        Fitting model: "linear" (Gaussian + linear background) or "poly2d" (Gaussian + quadratic background)
    gauss_fit_region : int, default = 20
        Pixels around correlation peak used for Gaussian fitting
    show_correlation_plot : bool, default = True
        If True, display correlation plots
    graph_dump : bool, default = False
        If True, save correlation plots to disk
    graph_dump_loc : str, optional
        Directory path for saved plots

    Returns
    ----------
    list of [analysis_variable, peak_fwhms, peak_fwhms_errs]
        analysis_variable : list of floats
            Independent variable values for successfully analyzed datasets
            (matches input analysis_variable, excluding failed correlations)
        peak_fwhms: list of floats
            Correlation peak FWHMS in pixels (spectral line broadening metric)
        peak_fwhms_errs: list of floats
            Standard errors on peak_fwhms from Gaussian fit covariance matrix

    Notes
    ----------
    If correlation peak detection or fitting fails for a dataset, NaN values
    are returned for that measurement and a UserWarning is issued.
    '''

    # ----- Preamble -----
    # Gather data from all the datasets pertaining to the order being cross-correlated
    corr_ord_datasets = []
    for i, df in enumerate(datasets):
        try:
            filtered = df.loc[[ord_to_corr]].reset_index(drop = True)
            corr_ord_datasets.append(filtered)
        except KeyError:
            warnings.warn(f"Order {ord_to_corr} not found in dataset {i}")

    if len(corr_ord_datasets) == 0:
        raise ValueError(f"Order {ord_to_corr} not found in any dataset")
    
    if dataset_base_corr >= len(corr_ord_datasets):
        raise IndexError(f"Base correlation index {dataset_base_corr} out of range")

    # Dataset to correlate against
    cross_base_spectra_y = corr_ord_datasets[dataset_base_corr]["spectra"][0][1]
    
    # Array for measured fwhms
    peak_fwhms, peak_fwhms_errs, analysis_variables = [], [], []

    # If True, set plot figure basic parameters
    if show_correlation_plot:
        corrs, corr_lags = [], []


    # ----- Main functionality -----
    # Cross-correlation
    for i in range(len(corr_ord_datasets)):
        dataset = corr_ord_datasets[i]
        spectrum_y = dataset['spectra'][0][1]

        # Cross-correlation function from scipy.signal
        correlation = correlate(
            cross_base_spectra_y,
            spectrum_y,
            method  = correlation_method,
            mode    = correlation_mode
        )
        
        # Normalize correlation to 1
        correlation /= np.max(correlation)
    
        # Cross-correlation lags function from scipy.signal
        correlation_lag = correlation_lags(
            len(spectrum_y),
            len(cross_base_spectra_y)
        )
        
        # Finding central peak from cross correlation
        corr_lag_cen = len(correlation_lag) // 2
        left, right = corr_lag_cen - gauss_fit_region, corr_lag_cen + gauss_fit_region

        # Try to detect peaks
        try:
            peaks, _ = find_peaks(correlation[left:right], prominence = 3 * np.median(correlation[left:right]))

            # If no peaks detected raise UserWarning
            if peaks.size == 0:
                raise UserWarning("Failed to detect any peaks.")
            
        except UserWarning as e:
            peak_fwhms.append(np.nan)
            peak_fwhms_errs.append(np.nan)
            analysis_variables.append(analysis_variable[i])

            warnings.warn(e)
            continue

        if use_with == 'linear':
            fit_function = _gaussian_with_linear
            init_guess = [correlation[left + peaks[0]], correlation_lag[left + peaks[0]], 0.1, 0.001, 1.]

        if use_with == 'poly2d':
            fit_function = _gaussian_with_2d
            init_guess = [correlation[left + peaks[0]], correlation_lag[left + peaks[0]], 0.1, 1., 1., 1.]

        # Try to fit gaussian with linear
        popt, pcov = curve_fit(
            fit_function,
            correlation_lag[left:right],
            correlation[left:right],
            p0 = init_guess
        )
        
        # Get variance from fit
        var = popt[2]
        var_err = np.sqrt(np.diag(pcov))[2]

        # Calculate full-width at half-maximum from variance
        fwhm = np.sqrt(8 * np.log(2) * var)
        fwhm_err = np.sqrt(4 * np.log(2))/np.sqrt(var) * var_err

        # If True, save correlation datas to array
        if show_correlation_plot:
            corrs.append(correlation)
            corr_lags.append(correlation_lag)

        peak_fwhms.append(fwhm)
        peak_fwhms_errs.append(fwhm_err)
        analysis_variables.append(analysis_variable[i])

    # If True, show correlation plot and check for graph dump
    if show_correlation_plot:
        fig, _ = simple_plot(
            corr_lags, 
            corrs, 
            (-gauss_fit_region, gauss_fit_region),
            xlabel = "Correlation lags",
            ylabel = "Correlation coefficient",
            title = f"Correlated order {ord_to_corr}"
        )

        # If True, save graphs to disk at specified location
        if graph_dump:

            #If graph dump location does not exist, create it
            if not os.path.isdir(graph_dump_loc):
                os.mkdir(graph_dump_loc)


            fig.savefig(os.path.join(graph_dump_loc, f"correlation_graph_order_{ord_to_corr}"))
        
        plt.show()
        plt.close(fig)

    # Return list of positions and measured fwhms
    return [analysis_variables, peak_fwhms, peak_fwhms_errs]


# V-curve fitting algorithm using Levenberg-Marquardt non-linear least squares method.
# Based on https://www.lost-infinity.com/v-curve-fitting-with-a-hyperbolic-function/
def v_curve_fitting(
        input_data_x            : np.ndarray,
        input_data_y            : np.ndarray,
        input_data_y_err        : np.ndarray,
        slope_start_distance    : int,
        current_order           : int,
        show_plot               : bool = True,
        graph_dump              : bool = False,
        graph_dump_loc          : str = 'graph_dump/vcurve'
    ):
    '''
    Fits hyperbolic V-curve to determine optimal spectrograph CCD-camera focuser position.

    Uses Levenberg-Marquardt non-linear least squares to fit a hyperbola to the
    relationship between camera focuser position and spectral line FWHM.
    The minimum of the fitted curve indicates the optimal focus position where
    spectral resolution is maximized.

    Mathematical form:
        H(x) = b * sqrt(1 + (x - c)² / a²) + d

        where:
        - x: camera focuser position
        - c: optimal focuser position (vertex x-coordinate)
        - d: minimum FWHM at optimal focus (vertex y-coordinate)
        - a, b: hyperbola shape parameters

        Note: the 'H' denotes the hyperbola model function (not the Hamiltonian)

    Algorithm:
        1. Remove NaN values from input data
        2. Identify initial minimum FWHM position
        3. Fit linear functions to left and right "wings" of V-curve
        4. Calculate intersection point of linear fits for initial guess params
        5. Use Levenberg-Marquadt to refine hyperbola parameters
        6. Estimate parameter uncertainties from Jacobian covariance matrix

    The Jacobian for residual minimization:
        J = [∂H/∂a, ∂H/∂b, ∂H/∂c, ∂H/∂d]

    The function is based on the following source:
    https://www.lost-infinity.com/v-curve-fitting-with-a-hyperbolic-function/
        
    Parameters
    ----------
    input_data_x : list of int
        Camera focuser positions
    input_data_y : list of float
        Measured spectral line FWHMS (pixels) at each position
    input_data_y_err : list of float
        Standard error on FWHM measurements (for visualization)
    slope_start_distance : int
        Number of points to exclude near center when fitting linear wings
        (avoids fitting lines to curved region near minimum)
    current_order : int
        Echelle order number (used for plot filename)
    show_plot : bool
        If True, display fitted V-curve with data points
    graph_dump : bool
        If True, save plot to disk
    graph_dump_loc : str
        Directory path for saved plots
    
    Results
    ----------
    list of [fit_params.x, fit_errors]
        fit_params : ndarray of shape(4,)
            Fitted hyperbola parameters [a, b, c, d]
            Key result: c is the optimal focus position
        fit_errors : ndarray of shape (4,)
            Standard errors on parameters from covariance matrix
    '''

    # ----- Input validation -----
    
    # Conversion to numpy arrays and ensuring float dtype
    input_data_x        = np.asarray(input_data_x, dtype = float)
    input_data_y        = np.asarray(input_data_y, dtype = float)
    input_data_y_err    = np.asarray(input_data_y_err, dtype = float)

    # Check for uniform array length
    if not (len(input_data_x) == len(input_data_y) == len(input_data_y_err)):
        raise ValueError(
            f"Input arrays must have the same length:"
            f"x = {len(input_data_x)}, y = {len(input_data_y)}, y_err = {len(input_data_y_err)}"
        )
    
    # Remove NaN-s from input data
    valid_mask = ~(np.isnan(input_data_x) | np.isnan(input_data_y) | np.isnan(input_data_y_err))

    input_data_x        = input_data_x[valid_mask]
    input_data_y        = input_data_y[valid_mask]
    input_data_y_err    = input_data_y_err[valid_mask]

    # Check for enough data points
    if len(input_data_x) < 5:
        raise ValueError(f"Not enough valid data points left after NaN removal: {len(input_data_x)}")

    # Validate slope_start_distance
    if slope_start_distance >= (len(input_data_x) // 2):
        raise ValueError(
            f"slope_start_distance {slope_start_distance} must be less than "
            f"half the number of data points ({len(input_data_x) // 2})"
        )
    
    # Initial minimum fwhm
    initial_minimum = np.argmin(input_data_y)

    # Warn if initial minimum is close to data boundary (characteristic of poor V-curve)
    if initial_minimum == 0 or initial_minimum == len(input_data_x) - 1:
        warnings.warn(
            f"Minimum FWHM is at data boundary (index {initial_minimum})."
            f"Data may not represent a proper V-curve and the function can produce faulty results."
            f"Consider expanding curve x-axis range",
            UserWarning
        )


    # ----- Main functionality -----

    # Left slope fitting
    popt, _ = curve_fit(
        _left_slope,
        input_data_x[:initial_minimum - slope_start_distance],
        input_data_y[:initial_minimum - slope_start_distance],
        p0 = [1, 1, input_data_x[initial_minimum], input_data_y[initial_minimum]]
    )
    a1, b1, c1, d1 = popt

    # Right slope fitting
    popt, _ = curve_fit(
        _right_slope,
        input_data_x[initial_minimum + slope_start_distance:],
        input_data_y[initial_minimum + slope_start_distance:],
        p0 = [1, 1, input_data_x[initial_minimum], input_data_y[initial_minimum]]
    )
    a2, b2, c2, d2 = popt

    # Left and right slope intersections
    x_intersect = (b1/abs(a1) * c1 + b2/abs(a2) * c2 + d1 - d2) / (b1/abs(a1) + b2/abs(a2))
    y_intersect_left = - b1/abs(a1) * (x_intersect - c1) + d1
    y_intersect_right = b2/abs(a2) * (x_intersect - c2) + d2

    # Initial guesses
    d_guess = (abs(y_intersect_left) + abs(y_intersect_right)) / 2
    b_guess = np.max(input_data_y) - d_guess
    a_guess = (np.max(input_data_x) - np.min(input_data_x)) / 4
    c_guess = input_data_x[initial_minimum]
    guess = [a_guess, b_guess, c_guess, d_guess]

    # Fit using Levenberg-Marquardt with Jacobian
    result = least_squares(
        fun = _residuals,
        x0 = guess,
        args = (input_data_x, input_data_y),
        jac = _jacobian,
        method = 'lm',
        max_nfev = 2000
    )

    # Error estimation using Jacobian sensitivity
    J = result.jac
    residual_var = np.var(result.fun)

    # Covariance matrix
    pcov = residual_var * np.linalg.inv(J.T @ J)
    result_errs = np.sqrt(np.diag(pcov))

    # If True, show plot containing data, V-curve, left and right slope fits, and other indicators.
    if show_plot:
        fig, _ = _plot_v_curve(
            input_data_x,
            input_data_y,
            np.array(input_data_y_err),
            initial_minimum,
            (a1, b1, c1, d1),
            (a2, b2, c2, d2),
            x_intersect,
            d_guess,
            result.x,
            y_lims = (0, None),
            current_order = current_order
        )

        #If True, save graphs to disk at specified location
        if graph_dump:

            #If graph dump location does not exist, create it
            if not os.path.isdir(graph_dump_loc):
                os.mkdir(graph_dump_loc)

            fig.savefig(os.path.join(graph_dump_loc, f"vcurve_graph_order_{current_order}"))

        plt.show()
        plt.close(fig)


    return [result.x, result_errs]


# ====================== UTILITY FUNCTIONS ======================
# ----- Public utility -----
def simple_plot(
        x_data  : list  | np.ndarray, 
        y_data  : list  | np.ndarray,
        x_lims  : tuple | None = None,
        y_lims  : tuple | None = None,
        figsize : tuple | None = None,
        xlabel  : str   | None = None,
        ylabel  : str   | None = None,
        title   : str   | None = None
    ) -> tuple:
    '''
    Simple line plot for quick visualisation of results.

    Supports both single datasets and multiple datasets (passed as list of arrays)

    Parameters
    ----------
    x_data : np.ndarray or list of np.ndarray
        X-axis data
    y_data : np.ndarray or list of np.ndarray
        Y-axis data. Must match x_data structure
    x_lims : tuple | None, default = None
        X-axis limits
    y_lims : tuple | None, default = None
        Y-axis limits
    figsize : tuple | None, default = None
        Figure size in inches. If None, uses module defaults (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT)
    xlabel : str | None, default = None
        X-axis label
    ylabel : str | None, default = None
        Y-axis label
    title : str | None, default = None
        Plot title
        
    Returns
    ----------
    plot_objects : tuple of (fig, ax)
        Matplotlib figure and axes objects

    Examples
    ----------
    Single dataset:
    >>> fig, ax = _simple_plot(np.array([1, 2, 3]), np.array([4, 5, 6]))

    Multiple datasets:
    >>> x = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> y = [np.array([7, 8, 9]), np.array([10, 11, 12])]
    >>> fig, ax = _simple_plot(x, y, xlabel = "Position", ylabel = "FWHM")
    '''

    # ----- Input validation -----
    is_multi = isinstance(x_data, list) and len(x_data) > 0 and hasattr(x_data[0], '__iter__')

    if is_multi:
        # Multiple datasets, validate lengths
        if len(x_data) != len(y_data):
            raise ValueError(
                f"When passing list of np.ndarray the datasets must be paired; 'x_data' and 'y_data' must be the same length"
            )
    
        # Validate pair lengths
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            if len(x) != len(y):
                raise ValueError(
                    f"Dataset{i}: subset pair 'x' and 'y' must be the same length"
                )
    else:
        # Single dataset length validation
        if len(x_data) != len(y_data):
            raise ValueError(
                f"'x_data' and 'y_data' must have the same length"
            )
        
    if figsize is None:
        figsize = (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT)
    

    # ----- Main functionality -----
    # Figure creation
    fig, ax = plt.subplots(
        figsize = figsize,
        layout = 'compressed'
    )
    
    # Check for multi-dataset and that it is actually more than 1 dataset
    if is_multi and len(x_data) > 1:
        # Use colormap for multiple datasets
        lin_colors = plt.cm.magma(np.linspace(0.2, 0.9, len(x_data)))

        for i in range(len(x_data)):
            ax.plot(x_data[i], y_data[i], color = lin_colors[i])
    else:
        # Handle both array and single-element list formats
        x = x_data[0] if isinstance(x_data, list) else x_data
        y = y_data[0] if isinstance(y_data, list) else y_data
        
        ax.plot(x, y)

    if x_lims:
        ax.set_xlim(*x_lims)
    
    if y_lims:
        ax.set_ylim(*y_lims)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    

    return fig, ax


# ----- Private helper functions -----
def _gaussian_centroid(
        y_data : np.ndarray
    ) -> np.ndarray:

    x_data = np.arange(0, len(y_data))

    popt, _ = curve_fit(_gaussian, x_data, y_data, p0 = [max(y_data), x_data[np.argmax(y_data)], 0.1])

    x_fit = np.arange(0, len(y_data), 0.1)
    y_fit = _gaussian(x_fit, *popt)

    peaks, _ = find_peaks(y_fit, prominence = 1)

    return x_fit[peaks]

def _spectre_mask(
        x               : np.ndarray, 
        order_low       : int, 
        order_center    : int, 
        order_high      : int, 
        trace_coeffs    : np.ndarray
    ) -> np.ndarray:
    '''
    Creates curved extraction mask for echelle order.

    Returns array of shape (n_rows, n_x_pixels) containing y-coordinates
    for extraction at each x-position.
    '''

    order_spine = np.polyval(trace_coeffs, x)

    lower_offset = order_low - order_center
    upper_offset = order_high - order_center

    n_rows = upper_offset - lower_offset + 1
    y_offsets = np.linspace(lower_offset, upper_offset, n_rows, dtype = int)

    # Broadcasting: spine is (n_x, ), y_offsets is (n_y, )
    # Result is (n_y, n_x) array
    mask = (order_spine[np.newaxis, :] + y_offsets[:, np.newaxis]).astype(int)

    return mask

def _get_ordernums(
        order_centers   : np.ndarray, 
        ref_order       : int,
        ref_order_pix   : int
    ) -> list[int]:
    '''
    Assings echelle order numbers based on a reference order position.

    Calculates order numbers for all detected orders by measuring their
    distance from a known reference order.

    Parameters
    ----------
    order_centers : np.ndarray
        Y-pixel positions of detected order centers
    ref_order : int
        Known order number of the reference order (e.g. 52)
    ref_order_pix : int
        Y-pixel position of the reference order

    Returns
    ----------
    order_numbers : list of int
        Order numbers for each detected order
    
    Example
    ----------
    If order_centers = [800, 1000, 1220, 1400] and ref_order = 52 at ref_order_pix = 1220,
    returns [54, 53, 52, 51]
    '''

    ref_order_idx = np.argmin(np.abs(order_centers - ref_order_pix))
    order_numbers = [ref_order + (ref_order_idx - i) for i in range(len(order_centers))]

    return order_numbers


# ----- Private visualisation functions -----
def _plot_v_curve(
        x_data          : np.ndarray,
        y_data          : np.ndarray,
        y_err           : np.ndarray,
        initial_min_idx : int,
        lslope_params   : tuple,
        rslope_params   : tuple,
        x_intersect     : float,
        d_guess         : float,
        fit_result      : tuple,
        x_lims          : tuple | None = None,
        y_lims          : tuple | None = None,
        current_order   : int   | None = None
    ) -> tuple:
    '''
    Plot V-curve fitting diganostic results.

    Creates a diagnostic plot showing:
    - Data points with error bars
    - Linear tangent fits to V-curve
    - Slope intersection point (initial guess)
    - Initial minimum position
    - Fitted hyperbola
    - Optimal focuser position (vertical line)

    Parameters
    ----------
    x_data : ndarray
        Analysis variable on which FWHMs depend
    y_data : ndarray
        Measured FWHMs
    y_err : ndarray
        FWHM measurement errors (for 95% CI error bars)
    initial_min_idx : int
        Index of initial minimum FWHM
    lslope_params : tuple of (a, b, c, d)
        Left slope parameters
    rslope_params : tuple of (a, b, c, d)
        Right slope parameters
    x_intersect : float
        X-coordinate of slope intersection
    d_guess : float
        Y-coordinate of slope intersection (initial minimum guess)
    fit_result : tuple of (a, b, c, d)
        Fitted hyperbola parameters
    x_lims : tuple of (min, max), optional
        X-axis limits
    y_lims : tuple of (min, max), optional
        Y-axis limits
    current_order : int or str, optional
        Order number for plot title

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects
    '''


    # Create an increased resolution x-axis dataset for smoother fitted function plotting
    x_smooth = np.linspace(min(x_data), max(x_data), 100)

    # ----- Main functionality -----
    # Figure creation
    fig, ax = plt.subplots(
        figsize = (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT),
        layout = 'compressed'
    )

    # Plot V-curve data
    ax.errorbar(
        x = x_data,
        y = y_data,
        yerr = 1.96 * y_err, # k = 1.96 for 95% confidence interval
        fmt = 'D', markersize = 4, capsize = 2, capthick = 0.25, mew = 0.5,
    )
    # Plot V-curve left wing linear regression
    ax.plot(
        x_smooth,
        _left_slope(x_smooth, *lslope_params),
        color = 'red', ls = '--',
    )
    # Plot V-curve right wing linear regression
    ax.plot(
        x_smooth,
        _right_slope(x_smooth, *rslope_params),
        color = 'red', ls = '--'
    )
    # Plot V-curve minimum location guess from wing linear regression intersection
    ax.scatter(
        x_intersect,
        d_guess,
        fc = 'none', edgecolor = 'red', marker = 'o', s = 64, lw = 1.
    )
    # Plot V-curve initial minimum guess
    ax.scatter(
        x_data[initial_min_idx],
        y_data[initial_min_idx],
        fc = 'none', edgecolor = 'orange', marker = 'o', s = 64, lw = 1.
    )
    # Plot hyperbolic curve using least-squares results
    ax.plot(
        x_smooth,
        _H(fit_result, x_smooth),
        color = 'seagreen', lw = .75, ls = '--'
    )
    # Plot vertical line indicating hyperbolic curve minimum
    ax.vlines(
        x = fit_result[2],
        ymin = 0,
        ymax = y_data[initial_min_idx] * 1.5,
        color = 'seagreen', ls = "--", lw = .75
    )


    if x_lims:
        ax.set_xlim(*x_lims)
    if y_lims:
        ax.set_ylim(*y_lims)
    if current_order:
        ax.set_title(f"Order {current_order}")

    return fig, ax


# ----- Private mathematical functions -----
def _gaussian(
        x   : np.ndarray, 
        a   : float, 
        mu  : float, 
        var : float
    ) -> np.ndarray:
    '''
    Gaussian (normal) distribution
    '''

    return (a / np.sqrt(2 * np.pi * var)) * np.exp(- (x - mu)**2 / (2 * var))

def _gaussian_with_linear(
        x           : np.ndarray,
        a           : float,
        mu          : float,
        var         : float,
        slope       : float, 
        intercept   : float
    ) -> np.ndarray:
    '''
    Gaussian with linear background
    '''

    gaussian = _gaussian(x, a, mu, var)
    linear = slope * x + intercept

    return gaussian + linear

def _gaussian_with_2d(
        x           : np.ndarray,
        a           : float, 
        mu          : float, 
        var         : float, 
        poly_a      : float, 
        poly_b      : float, 
        poly_c     : float
    ) -> np.ndarray:
    '''
    Gaussian with quadratic background
    '''
    gaussian = _gaussian(x, a, mu, var)
    poly_2d = poly_a * x**2 + poly_b * x + poly_c

    return gaussian + poly_2d

def _left_slope(
        x   : np.ndarray, 
        a   : float, 
        b   : float, 
        c   : float, 
        d   : float
    ) -> np.ndarray:
    '''
    Left tangent line to hyperbola V-curve
    '''

    return -(b/abs(a)) * (x-c) + d

def _right_slope(
        x   : np.ndarray, 
        a   : float, 
        b   : float, 
        c   : float, 
        d   : float
    ) -> np.ndarray:
    '''
    Right tangent line to hyperbola V-curve
    '''

    return (b/abs(a)) * (x-c) + d

def _H(
        params, 
        x
    ) -> np.ndarray:
    '''
    Hyperbola function for V-curve fitting.

    H(x) = n * sqrt(1 + (x-c)² / a²) + d

    Vertex at (c, d), shape parameters a and b
    '''

    a, b, c, d = params
    return b * np.sqrt(1 + (x - c)**2 / a**2) + d

def _residuals(
        params, 
        x, 
        y
    ) -> np.ndarray:
    '''
    Residuals for hyperbola least-squares fitting
    '''

    return y - _H(params, x)

def _jacobian(
        params, 
        x,
        y
    ) -> np.ndarray:
    '''
    Jacobian matrix for hyperbola parameter optimization
    '''

    a, b, c, d = params
    denom = np.sqrt(1 + (x - c)**2 / a**2)

    # Derivatives
    dH_da = -b * (x - c)**2 / (a**3 * denom)
    dH_db = denom
    dH_dc = -b / a**2 * (x - c) / denom
    dH_dd = np.ones_like(x)

    # Jacobian residuals
    J = np.column_stack((-dH_da, -dH_db, -dH_dc, -dH_dd))

    return J


# ====================== NAMESPACE CLEANUP ======================
# Prevent namespace pollution, remove imports from public interface

del np, pd, plt
del warnings, os
del find_peaks, peak_widths, correlate, correlation_lags
del curve_fit, least_squares