# Whoppsel: A toolkit for automated echelle-spectrograph data analysis

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Python module/toolkit that contains functions for automated data
analysis of echelle-spectrograph image data, developed for characterizing
temperature-dependent performance of the Whoppshel echelle-spectrograph.

Originally created for my BSc thesis in Physics at Tartu Observatory,
University of Tartu. Used for analysing thermal effects on echelle
order and spectral line positions on 2D spectrographic images.

## Overview

The module contains functions that are capable of detecting, tracing,
and extracting echelle orders with minimal intervention.

Further, a robust cross-correlation analysis function is included for
correlating orders over several datasets to quantify data quality metrics
and assess the effects of temperature on gathered image data quality.

A hyperbolic curve fitting function (v_curve_fitting) is also included
for optimal camera focuser position determination.

### Key Capabilities
- **Automated Order Detection**: Identifies echelle-spectra orders in 2D CCD images using vertical slice analysis and Gaussian fitting
- **Order Tracing**: Maps curved order positions across the detector using iterative polynomial fitting
- **Spectral Extraction**: Converts 2D echelle orders to calibrated 1D spectra
- **Cross-Correlation Analysis**: Measures spectral line broadening (FWHM) across experimental conditions. Can also be used to
                                  measure pixel-shifts in spectral line or echelle-order peak positions.
- **V-Curve Fitting**: Determines optimal focuser positions using hyperbolic curve fitting with Levenberg-Marquardt optimization

### Notes
- The default values for functions are heavily based on the echelle-spectrograph at Tartu Observatory, pertaining to the unique
  circumstances and different factors involved in the final spectrographic images.
- When performing cross-correlation analysis for focuser position determination, it is best to use the most in-focus images
  as the base for correlation.
- V-curve fitting is heavily reliant on having enough data points further away from the region around optimal focuser position,
  as these regions are used for tangent line fitting.

## Installation

### Requirements
```bash
pip install numpy pandas scipy matplotlib
```

Or use the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.7+
- numpy
- pandas
- scipy
- matplotlib

## Usage

### Basic Pipeline Workflow
```python
import Whoppsel as wpl
from astropy.io import fits

# Load pre-processed spectrographic images (FITS format)

# For order detection and tracing the usage of flat images is encouraged
# due to the echelle-orders having a continous and higher signal.

# Calibration images (for example using a ThAr spectral lamp) should be used
# for order extraction and further data analysis.

image_data = fits.open('ThAr_spectrum.fits')[0].data

imagedata_flat = fits.open('Flat_image.fits')[0].data
imagedata_calib = fits.open('Calibration_image.fits')[0].data

# Step 1: Detect echelle orders
detected_orders = wpl.order_detector(imagedata_flat)

# Step 2: Trace order positions across detector
traced_orders = wpl.order_tracer(imagedata_flat, detected_orders)

# Step 3: Extract 1D spectra
extracted_orders = wpl.order_extractor(imagedata_calib, traced_orders)

# Step 4: Analyze multiple datasets (e.g., at different temperatures)
datasets = [ext_ords_1, ext_ords_2, ext_ords_3, ...]
positions = [40000, 45000, 50000, ...]  # Focuser positions

results = wpl.cross_correlate(
    datasets           = datasets,
    analysis_variable = positions,
    dataset_base_corr = 0,
    ord_to_corr       = 52
)

# Step 5: Fit V-curve to find optimal focus
analysis_vars, fwhms, fwhm_errors = results
optimal_params, param_errors = wpl.v_curve_fitting(
    input_data_x           = analysis_vars,
    input_data_y           = fwhms,
    input_data_y_err       = fwhm_errors,
    slope_start_distance   = 2,
    current_order = 52
)

print(f"Optimal focus position: {optimal_params[2]:.2f} ± {param_errors[2]:.2f}")
```

## Core Functions

### `order_detector(image_data, ...)`
Detects echelle orders in 2D spectrograph images using vertical slice analysis and Gaussian peak fitting.

**Returns**: DataFrame with order numbers, centers, edges, and FWHMs

### `order_tracer(image_data, order_locs, ...)`
Traces echelle order curvature across the CCD using iterative box-window scanning and polynomial fitting.

**Returns**: Updated DataFrame with initial and refined trace polynomial coefficients

### `order_extractor(image_data, order_locs, ...)`
Extracts 1D spectra from 2D echelle orders using traced positions.

**Returns**: Updated DataFrame with extracted `[x_data, y_data]` spectra for each order

### `cross_correlate(datasets, analysis_variable, ...)`
Measures changes in 1D-spectra (spectral line broadening, spectral line pixel shifts) via cross-correlation between datasets.

**Returns**: `[analysis_positions, peak_fwhms, fwhm_errors]`

### `v_curve_fitting(input_data_x, input_data_y, ...)`
Fits hyperbolic V-curve to determine optimal focuser position using Levenberg-Marquardt non-linear least squares minimization.

**Returns**: `[fit_parameters, parameter_errors]` where `fit_parameters[2]` is the optimal focus position

## Scientific Background

This toolkit was developed in conjuction with the study on how temperature affects the quality of data collected from
an echelle-spectrograph. Key findings from the thesis:

- Temperature changes cause measurable focal plane shifts due to thermal expansion
- Cross-correlation analysis enables sub-pixel precision in detecting these shifts
- The relationship between focus position and spectral resolution follows a hyperbolic V-curve

## File Structure
```
Whoppsel.py    # Main pipeline functions
├── order_detector()         # Order detection
├── order_tracer()           # Order tracing
├── order_extractor()        # Spectral extraction
├── cross_correlate()        # Cross-correlation analysis
├── v_curve_fitting()        # Focus optimization
└── Utility functions        # Gaussian fitting, masking, etc.
```

## Limitations & Future Work

- Currently optimized for the Tartu Observatory echelle spectrograph (instrument-specific parameters)
- Wavelength calibration functionality exists in a separate modified version
- Could benefit from modular refactoring with configuration file support

## Citation

If you use this code in your research, please cite:
```
Karl Nõupuu (2025). "Tartu Observatooriumi ešell-spektrograafi võimekuse karakteriseerimine"
Bakalaureusetöö, Tartu Ülikool.
```

## License

MIT License - feel free to use and modify for your research

## Author

Developed by Karl Nõupuu as part of BSc research at the University of Tartu Observatory's stellar physics department.

Contact: karl.noupuu@hotmail.com

## Acknowledgments

- Tartu Observatory for providing access to the echelle spectrograph
- Tõnis Eenmäe for supervision and guidance
