import numpy as np

from extract import get_times, get_oppositions
from fit import bestMarsOrbitParams
from plot import plot


if __name__ == "__main__":

    data = np.genfromtxt(
        "../data/01_data_mars_opposition_updated.csv",
        delimiter=",",
        skip_header=True,
        dtype="int",
    )

    times = get_times(data)
    assert len(times) == 12, "times array is not of length 12"

    oppositions = get_oppositions(data)
    assert len(oppositions) == 12, "oppositions array is not of length 12"

    r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(
        times, oppositions
    )

    assert max(list(map(abs, errors))) == maxError, "maxError is not computed properly!"
    print(
        "Fit parameters: r = {:.4f}, s = {:.4f}, c = {:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(
            r, s, c, e1, e2, z
        )
    )
    print("The maximum angular error = {:2.4f}".format(maxError))

    # Plot the resulting best fit
    # plot(c, r, e1, e2, z, s, times, oppositions)