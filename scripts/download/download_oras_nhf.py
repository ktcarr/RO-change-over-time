import cdsapi
import numpy as np

if __name__ == "__main__":

    ## note: 2015 onwards is operational
    # product_type = "consolidated"
    # yrs = [str(y) for y in np.arange(1958,1978)]
    # yrs = [str(y) for y in np.arange(1978,1999)]
    # yrs = [str(y) for y in np.arange(1999,2015)]

    product_type = "operational"
    yrs = [str(y) for y in np.arange(2015, 2021)]

    dataset = "reanalysis-oras5"
    request = {
        "product_type": [product_type],
        "vertical_resolution": "single_level",
        "variable": ["net_downward_heat_flux"],
        "year": yrs,
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()
