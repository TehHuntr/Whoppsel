# Whoppsel: Echelle Spectrograph Analysis Pipeline

Python pipeline for automated analysis of echelle spectrograph data,
developed for characterizing temperature-dependent performance of
the Whoppsel echelle-spectrograph.

Originally created for my BSc thesis in Physics at the University of Tartu,
analysing thermal effects on spectral line positions and focus quality
in the Tartu Observatory's echelle-spectrograph.

## Overview

This pipeline processes raw 2D CCD spectrograph images through automated
order detection, tracing, extraction, and cross-correlation analysis
to measure instrumental performance characteristics.

### Key Capabilities
- **Automated Order Detection**: Indentifies echelle-spectra orders in 2D CCD images using vertical slice analysis and Gaussian fitting
- **Order Tracing**: Maps curved order positions across the detector using iterative polynomial fitting
- **Spectral Extraction**: Converts 2D echelle orders to calibrated 1D spectra
- **Cross-Correlation Analysis**: Measures spectral line broadening (FWHM) across experimental conditions
- **V-Curve Fitting**: Determines optimal focuser positions using hyperbolic curve fitting with Levenberg-Marquardt optimization

## Installation

### Requirements
```bash
pip install numpy pandas scipy astropy matplotlib
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
- astropy
- matplotlib

## Usage

### Basic Pipeline Workflow
```python
from Whoppsel import (
    order_detector, 
    order_tracer, 
    order_extractor,
    cross_correlate,
    v_curve_fitting
)
from astropy.io import fits


# Load raw spectrograph data (FITS format)
image_data = fits.open('ThAr_spectrum.fits')[0].data

# Step 1: Detect echelle orders
orders = order_detector(image_data)

# Step 2: Trace order positions across detector
orders = order_tracer(image_data, orders)

# Step 3: Extract 1D spectra
orders = order_extractor(image_data, orders)

# Step 4: Analyze multiple datasets (e.g., at different temperatures)
datasets = [orders1, orders2, orders3, ...]
temperatures = [15.2, 18.5, 21.3, ...]  # °C

results = cross_correlate(
    datasets=datasets,
    analysis_variable=temperatures,
    dataset_base_corr=0,
    order_to_correlate=52
)

# Step 5: Fit V-curve to find optimal focus
positions, fwhms, fwhm_errors = results
optimal_params, param_errors = v_curve_fitting(
    input_data_x=positions,
    input_data_y=fwhms,
    input_data_y_err=fwhm_errors,
    slope_start_distance=2,
    current_order=52
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
Measures spectral line broadening via cross-correlation between datasets.

**Returns**: `[analysis_positions, peak_fwhms, fwhm_errors]`

### `v_curve_fitting(input_data_x, input_data_y, ...)`
Fits hyperbolic V-curve to determine optimal focus position using Levenberg-Marquardt optimization.

**Returns**: `[fit_parameters, parameter_errors]` where `fit_parameters[2]` is the optimal focus position

## Scientific Background

This pipeline was developed to study how temperature variations affect an echelle spectrograph's optical alignment. Key findings from the thesis work:

- Temperature changes cause measurable focal plane shifts due to thermal expansion
- Cross-correlation analysis enables sub-pixel precision in detecting these shifts
- The relationship between focus position and spectral resolution follows a hyperbolic V-curve

## File Structure
```
spectrograph_pipeline.py    # Main pipeline functions
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
Karl Nõupuu (2026). "Tartu Observatooriumi ešell-spektrograafi võimekuse karakteriseerimine"
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
