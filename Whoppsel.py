try:
    import warnings
    import os
except ImportError as e:
    print(f"Failed importing Python standard library: {e}")
    quit()
finally:
    print(f"Import of Python standard libraries successful.")
## ====================== IMPORTING PIPELINE DEPENDENCIES ======================
try:
    # ---------------------- DATA MANIPULATION DEPENDENCIES ----------------------
    import numpy as np
    import pandas as pd
    # ---------------------- SCIPY DEPENDENCIES ----------------------
    from scipy.signal import find_peaks, peak_widths, correlate, correlation_lags
    from scipy.optimize import curve_fit, least_squares
    # ---------------------- ASTROPY DEPENDENCIES ----------------------
    from astropy.stats import sigma_clip
    # ---------------------- OTHER DEPENDENCIES ----------------------
    import matplotlib.pyplot as plt
    # ---------------------- INHOUSE DEPENDENCIES ----------------------
except ImportError as e:
    print(f"Failed importing pipeline dependency library: {e}")
    quit()
finally:
    print(f"Import of pipeline dependencies successful.")


def order_detector(image_data,
                   vert_slice_idx       : int   = 1024,
                   vert_slice_radius    : int   = 10,
                   max_signal           : int   = 6000000,
                   gauss_width_at       : float = 0.5,
                   order_pix52_loc      : int   = 1220):
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
    gauss_width_at : float, default = 0.5
        Relative height for (0 - 1.0) for Gaussian width measurement (0.5 = FWHM)
    order_pix52_loc : int, default = 1220
        Y-axis pixel location for order 52 (reference order for numbering)

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

    nx, ny = np.shape(image_data)

    # Checking for user defined parameter validity or UserErrors

    ## IndexError for out of bounds indexing
    if (vert_slice_idx > (nx - 1)) or (vert_slice_idx < 0):
        raise IndexError(f"x-axis index {vert_slice_idx} out of bounds for image shape [{0}:{nx - 1}, {0}:{ny - 1}]")
    
    ## ValueError for gaussian width
    if (gauss_width_at <= 0) or (gauss_width_at > 1.0):
        raise ValueError(f"Gaussian width percentage should fall into the range (0.0, 1.0]")
    
    ## Warning for vertical slice radius, that might result in a non uniform vertical slice of data
    if ((vert_slice_idx - vert_slice_radius) < 0) or ((vert_slice_idx + vert_slice_radius) > nx):
        warnings.warn(f"Defined vertical slice radius {vert_slice_radius} extends over"
                      f"image bounds (0:{nx}, 0:{ny}).", UserWarning)


    ## Defining a slice of the image used order location detection
    vert_slice = image_data[:,
                            max(0, vert_slice_idx - vert_slice_radius):
                            min(nx, vert_slice_idx + vert_slice_radius)]
    
    # ---------------------- SLICING ----------------------
    x_data = np.arange(0, 2048, 1)
    y_data = sum(np.transpose(vert_slice))

    # ---------------------- INITIAL PEAK DETECTION ----------------------
    initial_peaks, _ = find_peaks(y_data, prominence = 10. * np.median(y_data))
    initial_peaks = [i for i in initial_peaks if y_data[i] < max_signal]

    # ---------------------- SECONDARY PEAK DETECTION ----------------------
    peak_fwhms, peak_centers, peak_los, peak_his = [], [], [], []
    unrounded_peak_centers = []
    for peak in initial_peaks:
        peak_lo = peak - int(np.round(np.mean(np.diff(initial_peaks))) / 2)
        peak_hi = peak + int(np.round(np.mean(np.diff(initial_peaks))) / 2)
        peak_x_data = x_data[peak_lo:peak_hi]
        peak_y_data = y_data[peak_lo:peak_hi]
        peak_x_fit_data = np.arange(min(peak_x_data), max(peak_x_data), 0.01)

        # ---------------------- GAUSSIAN FITTING ----------------------
        try:
            popt, _ = curve_fit(gaussian, peak_x_data, peak_y_data, p0 = [y_data[peak], x_data[peak], 0.1])
        except:
            continue

        gaussian_fit = gaussian(peak_x_fit_data, *popt)
        refit_peaks, _ = find_peaks(gaussian_fit, prominence = 1)
        refit_fwhms = peak_widths(gaussian_fit, refit_peaks, gauss_width_at)

        peak_centers.append(int(np.round(refit_peaks[0]) / 100) + x_data[peak_lo])
        unrounded_peak_centers.append(peak_x_fit_data[refit_peaks[0]] + x_data[peak_lo])
        peak_fwhms.append(refit_fwhms[0][0] / 100)
        peak_los.append(x_data[peak_lo] + int(np.round(refit_fwhms[2][0]) / 100))
        peak_his.append(x_data[peak_lo] + int(np.round(refit_fwhms[3][0]) / 100))

    detected_orders = pd.DataFrame(data = {"order_fwhms"        : peak_fwhms,
                                           "order_low_edge"     : peak_los,
                                           "order_centers"      : peak_centers,
                                           "unround_cens"       : unrounded_peak_centers,
                                           "order_high_edge"    : peak_his})
    detected_orders.insert(0, "order_nums", _get_ordernums(detected_orders["order_centers"], order_pix52_loc))
    return detected_orders


def order_tracer(image_data,
                 order_locs             : pd.DataFrame,
                 trace_window_radius    : int = 10,
                 trace_poly_order       : int = 2,
                 trace_x_min            : int = 800,
                 trace_x_max            : int = 1100,
                 refine_count           : int = 3,
                 refine_poly_order      : int = 4,
                 refine_x_min           : int = 500,
                 refine_x_max           : int = 1500):
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


    order_centers = order_locs["order_centers"]
    order_poly_coeffs, refined_poly_coeffs = [], []

    x_range = np.arange(trace_x_min, trace_x_max, 1)
    nx, ny = np.shape(image_data)

    for center in order_centers:
        trace = []
        center = int(np.round(center))

        for x in x_range:
            hor_slice = slice(max(0, center - trace_window_radius),
                              min(ny, center + trace_window_radius))
            
            try:    y_rel = gaussian_centroid(image_data[hor_slice, x])
            except: y_rel = np.array([0])

            if y_rel.size == 0: y_abs = hor_slice.start + 0
            else:               y_abs = hor_slice.start + y_rel[0]

            trace.append(y_abs)
        
        trace_coeffs = np.polyfit(x_range, trace, trace_poly_order)
        order_poly_coeffs.append(trace_coeffs)

        for ref in range(refine_count):
            refined_trace = []
            guide_polynomial = np.polyval(trace_coeffs, np.arange(0, nx))
            
            xx_range = np.arange(max(0, refine_x_min - ref * 100),
                                 min(nx, refine_x_max + ref * 100), 10)
            
            for xx in xx_range:
                guide_center = int(np.round(guide_polynomial[xx]))
                hor_slice = slice(max(0, guide_center - int(trace_window_radius / 2)),
                                  min(ny, guide_center + int(trace_window_radius / 2)))
                
                try:    y_rel = gaussian_centroid(image_data[hor_slice, xx])
                except: y_rel = np.array([0])

                if y_rel.size == 0: y_abs = hor_slice.start + 0
                else:               y_abs = hor_slice.start + y_rel[0]

                refined_trace.append(y_abs)

            trace_coeffs = np.polyfit(xx_range, refined_trace, refine_poly_order)
        
        refined_poly_coeffs.append(trace_coeffs)

    order_locs["init_trace_coeffs"] = order_poly_coeffs
    order_locs["ref_trace_coeffs"] = refined_poly_coeffs
    return order_locs


def order_extractor(image_data,
                    order_locs,
                    ext_x_min : int = 500,
                    ext_x_max : int = 1500):
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

    order_centers   = order_locs["order_centers"]
    order_lows      = order_locs["order_low_edge"]
    order_highs     = order_locs["order_high_edge"]
    trace_coeffs    = order_locs["ref_trace_coeffs"]

    x_data = np.arange(ext_x_min, ext_x_max, 1)
    spectres = []

    for i in range(len(order_locs.index)):
        y_data = _spectra_mask(x_data,
                               order_centers[i],
                               order_lows[i],
                               order_highs[i],
                               trace_coeffs[i])
        
        try:
            spectre_datas = []
            for j in range(len(x_data)):
                spectre_datas.append(np.array([image_data[y, x] for x, y in zip(x_data, y_data[j])]))
        except:
            pass
    
        spectres.append([x_data, sum(spectre_datas)])

    order_locs["spectra"] = spectres
    return order_locs


# Wrapper function for _measure_fwhms, this is so that
# the measurement of fwhms of one spectra and all order spectra is
# separate
def measure_fwhms(order_locs : pd.DataFrame,
                 exp_res    : int = 5,
                 fit_width  : int = 10,
                 sigma_clipping : bool = True,
                 sigmas : int = 3,
                 sigma_func : str = "median"):
    
    spectra = order_locs["spectra"]

    peak_fwhms, peak_stds = [], []
    for spectre in spectra:
        measured_fwhms = _measure_fwhms(spectre,
                                        exp_res,
                                        fit_width,
                                        sigma_clipping,
                                        sigmas,
                                        sigma_func)
        peak_fwhms.append(np.mean(measured_fwhms))
        peak_stds.append(np.std(measured_fwhms))

    order_locs["mean_fwhms"] = peak_fwhms
    order_locs["fwhm_stds"] = peak_stds
    return order_locs

# _measure_fwhms method that handles taking a singular extracted
# spectra, detects peaks that are with prominence SNR / 2 and measures
# their fwhms.
def _measure_fwhms(spectre,
                  exp_res,
                  fit_width,
                  sigma_clipping,
                  sigmas,
                  sigma_func):

    peak_fwhm = []
    x_data, y_data = spectre
    SNR = sum(y_data) / np.sqrt(sum(y_data))

    peaks, _ = find_peaks(y_data,
                        prominence = SNR / 2,
                        distance = exp_res)
    for peak in peaks:
        min_idx, max_idx = max(0, peak - fit_width), min(len(x_data) - 1, peak + fit_width)
        x_range = np.arange(x_data[min_idx],
                            x_data[max_idx],
                            0.1)
        try:
            popt, _ = curve_fit(gaussian,
                                x_data[min_idx:max_idx],
                                y_data[min_idx:max_idx],
                                p0 = [y_data[peak], x_data[peak], 0.1])
        except:
            continue

        y_fit = gaussian(x_range, *popt)
        peaks, _ = find_peaks(y_fit, prominence = np.median(y_data))
        fwhm = peak_widths(y_fit, peaks, 0.5); fwhm = fwhm[0] / 10

        try:
            peak_fwhm.append(fwhm[0])
        except IndexError:
            continue

    if sigma_clipping is True:
        peak_fwhm = sigma_clip(peak_fwhm,
                                        sigma = sigmas,
                                        maxiters = None,
                                        cenfunc = sigma_func)
        
    return peak_fwhm
# ================================= UTILS =================================
def gaussian(x, a, mean, variance):
    return (a / np.sqrt(2 * np.pi * variance)) * np.exp(- (x - mean)**2 / (2 * variance))

def gaussian_with_linear(x, a, mu, var, slope, intercept):
    gaussian = a / np.sqrt(2 * np.pi * var) * np.exp(-(x - mu)**2 / (2 * var))
    linear = slope * x + intercept
    return gaussian + linear

def gaussian_with_2d(x, a, mu, var, b, c, d):
    gaussian = a / np.sqrt(2 * np.pi * var) * np.exp(-(x - mu)**2 / (2 * var))
    poly_2d = a*x**2 + b*x + c
    return gaussian + poly_2d

def gaussian_centroid(y_data):
    x_data = np.arange(0, len(y_data), 1)

    popt, _ = curve_fit(gaussian, x_data, y_data, p0 = [max(y_data), x_data[np.argmax(y_data)], 0.1])

    x_fit = np.arange(0, len(y_data), 0.1)
    y_fit = gaussian(x_fit, *popt)

    peaks, _ = find_peaks(y_fit, prominence = 1)

    return x_fit[peaks]


def _spectra_mask(x, order_center, order_low, order_high, trace_coeffs):
    spectre_lines = []
    spectre_spine = np.polyval(trace_coeffs, x)
    spectre_width = np.arange(np.ceil(order_low - order_center), np.ceil(order_high - order_center) + 2, 1)
    for pix in spectre_width:
        spectre_lines.append([int(i) for i in (spectre_spine + pix)])
    return spectre_lines


def _get_ordernums(order_centers, order_pix52_loc):
    loc_52 = np.argmin(np.abs(np.array(order_centers) - order_pix52_loc))
    return [52 + (loc_52 - i) for i in range(len(order_centers))]


def cross_correlate(datasets                : list,
                    analysis_variable       : list,
                    dataset_base_corr       : int,
                    order_to_correlate      : int,
                    correlation_method      : str = "fft",
                    correlation_mode        : str = "full",
                    use_with                : str = "linear",
                    gauss_fit_region        : int = 20,
                    show_correlation_plot   : bool = True,
                    graph_dump              : bool = False,
                    graph_dump_loc          : str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corr_graph_dump")):
    '''
    Measures spectral line broadening across experimental conditions via cross-correlation.
    
    Compares extracted 1D spectra to quantify changes in spectral line width (FWHM).
    Uses cross-correlation to measure pixel-level shifts and Gaussian
    fitting of the correlation peak to achieve sub-pixel precision.

    The correlation peak's FWHM indicates spectral line broadening, which reflects
    focus quality and instrumental stability across different conditions.

    Algorithm:
        1. Select specified order from all datasets
        2. Cross-correlate each spectrum agaisnt the reference spectrum (dataset_base_correlate)
        3. Fit Gaussian (with linear or polynomial background) to correlation peak
        4. Extract peak FWHM and position with uncertainties from covariance matrix
        5. Optionally plot correlation functions for quality control

    Parameters
    ----------
    datasets : list of pd.DataFrame
        List of order_locs DataFrames from order_extract(), one per 
    analysis_variable : list of list
        Independent variable values corresponding toe each dataset.
        Used to label measurements when analysing how spectral properties change with experimental conditions.
    dataset_base_corr : int
        Index of reference dataset for cross-correlation
    order_to_correlate : int
        Echelle order number to analyse
    correlation_method : str, default = "fft"
        scipy.signal.correlate method: "fft" (fast) or "direct"
    correlation_mode : str, default = "full"
        scipy.signal.correlate mode: "full", "valid", or "same"
    use_with : str, default = "linear"
        Fitting model: "linear" (Gaussian + linear background) or "poly2d" (Gaussian + quadratic background)
    gauss_fit_region : int, default = 20
        Pixels around correlation peak used for Gaussian fitting
    show_correlation_plot : bool, default = "True"
        If True, display correlation plots
    graph_dump : bool, default = "False"
        If True, save correlation plots to disk
    graph_dump_loc : str, optional
        Directory path for saved plots

    Returns
    ----------
    list of [analysis_variable, peak_fwhms, peak_fwhms_errs]
        analysis_variable:
            analysis_positions: list of floats
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
    # Narrow given dataset to specified order
    narrowed_datasets = []
    for i in range(len(datasets)):
        current_dataset = datasets[i].copy()
        narrowed_datasets.append(current_dataset[current_dataset["order_nums"].isin([order_to_correlate])].reset_index(drop = True))
    # Dataset to correlate against
    cross_base_spectra_y = narrowed_datasets[dataset_base_corr]["spectra"][0][1]
    # Array for measured fwhms
    peak_fwhms = []
    peak_fwhms_errs = []
    camera_positions = []
    # Cross correlate
    if show_correlation_plot:
        golden = (1 + 5 ** 0.5) / 2
        pt = 1 / 72.
        ltx_textwidth = 455.24408 * pt
        fig, ax = plt.subplots(1, 1, figsize = (ltx_textwidth, ltx_textwidth/golden), layout = "compressed")
        linear_colors = plt.cm.magma(np.linspace(0.2, 0.9, len(narrowed_datasets)))

    for i in range(len(narrowed_datasets)):
        dataset = narrowed_datasets[i]
        spectra_y = dataset["spectra"][0][1]
        correlation = correlate(cross_base_spectra_y,
                               spectra_y,
                               method = correlation_method,
                               mode = correlation_mode)
        correlation /= np.max(correlation)
        correlation_lag = correlation_lags(len(spectra_y),
                                          len(cross_base_spectra_y))
        
        if show_correlation_plot:
            ax.plot(correlation_lag, correlation, lw = .6, label = f"Järk {order_to_correlate}", color = linear_colors[i])
            ax.set_xlim(-gauss_fit_region, gauss_fit_region)
            ax.set_xlabel("Nihe dispersiooniteljes (piksel)")
            ax.set_ylabel("Korrelatsioonikoefitsient")
            #handles, labels = plt.gca().get_legend_handles_labels()
            #by_label = dict(zip(labels, handles))
            #ax.legend(by_label.values(), by_label.keys(), loc = "upper right")

        # Finding central peak from cross correlation
        left, right = int(len(correlation_lag) / 2 - gauss_fit_region), int(len(correlation_lag) / 2 + gauss_fit_region)
        # Try and detect peaks
        try:
            peaks, _ = find_peaks(correlation[left:right], prominence = 3 * np.median(correlation[left:right]))
            # If no peaks detected raise UserWarning
            if peaks.size == 0:
                raise UserWarning("Failed to detect any peaks.")
        except UserWarning as e:
            peak_fwhms.append(np.nan)
            peak_fwhms_errs.append(np.nan)
            camera_positions.append(analysis_variable[i])
            warnings.warn(e)
            continue
        if use_with == "linear":
            fit_function = gaussian_with_linear
            init_guess = [correlation[left + peaks[0]], correlation_lag[left + peaks[0]], 0.1, 0.001, 1.]
        if use_with == "poly2d":
            fit_function = gaussian_with_2d
            init_guess = [correlation[left + peaks[0]], correlation_lag[left + peaks[0]], 0.1, 1., 1., 1.]
        # Try to fit gaussian with linear
        popt, pcov = curve_fit(fit_function,
                        correlation_lag[left:right],
                        correlation[left:right],
                        p0 = init_guess)
        
        var = popt[2]
        var_err = np.sqrt(np.diag(pcov))[2]

        fwhm = np.sqrt(8 * np.log(2) * var)
        fwhm_err = np.sqrt(4 * np.log(2))/np.sqrt(var) * var_err

        peak_fwhms.append(fwhm)
        peak_fwhms_errs.append(fwhm_err)
        camera_positions.append(analysis_variable[i])
    if show_correlation_plot:
        if graph_dump:
            if not os.path.isdir(graph_dump_loc):
                os.mkdir(graph_dump_loc)

            plt.savefig(os.path.join(graph_dump_loc, f"ccorr_or_{order_to_correlate}.pdf"))

        plt.show()
    # Return list of positions and measured fwhms
    return [camera_positions, peak_fwhms, peak_fwhms_errs]


def v_curve_fitting(input_data_x : list,
                    input_data_y : list,
                    input_data_y_err : list,
                    slope_start_distance : int,
                    current_order : int,
                    show_plot : bool = True,
                    graph_dump : bool = False,
                    graph_dump_loc : str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcurve_graph_dump")):
    '''
    Fits hyperbolic V-curve to determine optimal spectrograph CCD-camera focuser position.

    Uses Levenberg-Marquardt non-linear least squares to fit a hyperbola to the
    relationship between camera focuser position and spectral line FWHM.
    The minimum of the fitted curve indicates the optimal focus position where
    spectral resolution is maximized.

    Physical model:
        H(x, a, b, c, d) = b * sqrt(1 + (x - c)² / a²) + d

        where:
        - x: camera focuser position
        - c: optimal focuser position (vertex x-coordinate)
        - d: minimum FWHM at optimal focus (vertex y-coordinate)
        - a, b: hyperbola shape parameters

    Algorithm:
        1. Remove NaN values from input data
        2. Identify initial minimum FWHM position
        3. Fit linear functions to left and right "wings" of V-curve
        4. Calculate intersection point of linear fits for inital guess params
        5. Use Levenberg-Marquadt to refine hyperbola parameters
        6. Estimate parameter uncertainties from Jacobian covariance matrix

    The Jacobian for residual minimization:
        J = [∂H/∂a, ∂H/∂b, ∂H/∂c, ∂H/∂d]

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
    input_data_x = np.array(input_data_x)[~np.isnan(np.array(input_data_y))]
    input_data_y_err = np.array(input_data_y_err)[~np.isnan(np.array(input_data_y))]
    input_data_y = np.array(input_data_y)[~np.isnan(np.array(input_data_y))]

    # Define initial minimal fwhm
    initial_minimum = np.argmin(input_data_y)
    # Left slope fitting
    popt, _ = curve_fit(left_slope,
                        input_data_x[:initial_minimum - slope_start_distance],
                        input_data_y[:initial_minimum - slope_start_distance],
                        p0 = [1, 1, input_data_x[initial_minimum], input_data_y[initial_minimum]])
    a1, b1, c1, d1 = popt
    # Right slope fitting
    popt, _ = curve_fit(right_slope,
                        input_data_x[initial_minimum + slope_start_distance:],
                        input_data_y[initial_minimum + slope_start_distance:],
                        p0 = [1, 1, input_data_x[initial_minimum], input_data_y[initial_minimum]])
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
        fun = residuals,
        x0 = guess,
        args = (input_data_x, input_data_y),
        jac = jacobian,
        method = "lm",
        max_nfev = 2000
    )
    # Error estimation using Jacobian sensitivity
    J = result.jac
    residual_var = np.var(result.fun)
    # Covariance matrix
    pcov = residual_var * np.linalg.inv(J.T @ J)
    result_errs = np.sqrt(np.diag(pcov))

    if show_plot:
        golden = (1 + 5 ** 0.5) / 2
        pt = 1 / 72.
        ltx_textwidth = 455.24408 * pt
        fig, ax = plt.subplots(1, 1, figsize = (ltx_textwidth, ltx_textwidth / golden))
        x_range = np.linspace(min(input_data_x), max(input_data_x), 100)
        #ax.scatter(input_data_x, input_data_y, s = 8, marker = "D")
        ax.errorbar(x = input_data_x,
                    y = input_data_y,
                    yerr = 1.96 * np.array(input_data_y_err), 
                    fmt = "D", 
                    markersize = 4, 
                    capsize = 2, 
                    capthick = 0.25, 
                    mew = 0.5,
                    lw = 0.75,
                    label = "Andmepunktid")
        ax.plot(x_range, left_slope(x_range, a1, b1, c1, d1), color = "red", ls = "--", lw = 1.25, label = "Hüperbooli puutujad")
        ax.plot(x_range, right_slope(x_range, a2, b2, c2, d2), color = "red", ls = "--", lw = 1.25)
        ax.scatter(x_intersect, d_guess, fc = "none", edgecolor = "red", marker = "o", s = 64, linewidths = 1, label = "Puutujate ristumiskoht")
        ax.scatter(input_data_x[initial_minimum], input_data_y[initial_minimum], fc = "none", edgecolors = "orange", marker = "o", s = 64, linewidths = 1, label = "Esmane minimaalne pool-laius")
        ax.plot(np.arange(min(input_data_x), max(input_data_x), 0.1), H(result.x, np.arange(min(input_data_x), max(input_data_x), 0.1)), color = "green", lw = 1.25, ls = "--", label = "Sobitatud hüperbool")
        ax.grid(color = "gray", ls = "--", lw = .5, alpha = .5)
        ax.legend(loc = "upper center")
        ax.vlines(x = result.x[2], ymin = 0, ymax = input_data_y[initial_minimum] * 1.5, color = "green", ls = "--", lw =1.25)
        ax.set_xlabel("Kaamera fokuseerija positsioon")
        ax.set_ylabel("Pool-laius korrelatsioonist (piksel)")
        ax.set_ylim(0)
        if graph_dump:
            if not os.path.isdir(graph_dump_loc):
                os.mkdir(graph_dump_loc)

            plt.savefig(os.path.join(graph_dump_loc, f"vcurve_or_{current_order}.pdf"))
        plt.show()

    return [result.x, result_errs]


def left_slope(x, a, b, c, d):
    return -(b/abs(a)) * (x-c) + d

def right_slope(x, a, b, c, d):
    return (b/abs(a)) * (x-c) + d

def H(params, x):
    a, b, c, d = params
    return b * np.sqrt(1 + (x - c)**2 / a**2) + d
# Defing the model residual function
def residuals(params, x, y):
    return y - H(params, x)

def jacobian(params, x, y):
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