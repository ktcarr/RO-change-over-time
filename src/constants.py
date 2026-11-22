"""
Global constants: region boundaries and physical constants.

Data and output folders are not here: they come from the DATA_FP and SAVE_FP
environment variables, read when a file is loaded or saved (see README.md).
"""

## longitude ranges (degrees east) for the western and eastern equatorial
## boxes used in the paper: T_w is averaged over LONS_W, T_e over LONS_E
LONS_W = [150, 190]


LONS_E = [240, 280]


## latitude range (degrees north) for "equatorial" averages
LATS_EQ = [-5, 5]

## seawater density (kg m^-3) and specific heat (J kg^-1 K^-1), for converting
## surface heat fluxes to mixed-layer warming rates
RHO_SW = 1.02e3
CP_SW = 4.2e3

## seconds per year, with 30-day months (as in the old scripts)
SEC_PER_YR = 12 * 30 * 8.64e4
