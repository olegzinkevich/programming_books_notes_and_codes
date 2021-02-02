# Interpolation
# Data that contains information usually has a functional form, and as analysts we want
# to model it. Given a set of sample data, obtaining the intermediate values between the
# points is useful to understand and predict what the datawill do in the non-sampled domain.
# SciPy offers well over a dozen different functions for interpolation, ranging from
# those for simple univariate cases to those for complex multivariate ones. Univariate
# interpolation is used when the sampled data is likely led by one independent variable,
# whereas multivariate interpolation assumes there is more than one independent
# variable.

# There are two basic methods of interpolation: (1) Fit one function to an entire dataset
# or (2) fit different parts of the dataset with several functions where the joints of each
# function are joined smoothly.The second type is known as a spline interpolation, which
# can be a very powerful tool when the functional form of data is complex.

# The example below interpolates a sinusoidal function (see Figure 3-6) using
# scipy.interpolate.interp1d with different fitting parameters. The first parameter is a
# “linear” fit and the second is a “quadratic” fit.

import numpy as np
from scipy.interpolate import interp1d

# Setting up fake data
x = np.linspace(0, 10 * np.pi, 20)
y = np.cos(x)

# Interpolating data
fl = interp1d(x, y, kind='linear')
fq = interp1d(x, y, kind='quadratic')
print(fl)
print(fq)

# x.min and x.max are used to make sure we do not
# go beyond the boundaries of the data for the
# interpolation.
xint = np.linspace(x.min(), x.max(), 1000)
yintl = fl(xint)
yintq = fq(xint)

print(yintl)
print(yintq)

# Can we interpolate noisy data? Yes, and it is surprisingly easy, using a spline-fitting
# function called scipy.interpolate.UnivariateSpline. (The result is shown in

from scipy.interpolate import UnivariateSpline
# Setting up fake data with artificial noise
sample = 30
x = np.linspace(1, 10 * np.pi, sample)
y = np.cos(x) + np.log10(x) + np.random.randn(sample) / 10
# Interpolating the data
# The option s is the smoothing factor, which should be used when fitting data with
# noise. If instead s=0, then the interpolation will go through all points while ignoring
# noise.
f = UnivariateSpline(x, y, s=1)
# x.min and x.max are used to make sure we do not
# go beyond the boundaries of the data for the
# interpolation.
xint = np.linspace(x.min(), x.max(), 1000)
yint = f(xint)

# Last but not least, we go over a multivariate example—in this case, to reproduce an
# image. The scipy.interpolate.griddata function is used for its capacity to deal with
# unstructuredN-dimensional data. For example, if you have a 1000× 1000-pixel image,
# and then randomly selected 1000 points, how well could you reconstruct the image

from scipy.interpolate import griddata

# Defining a function
ripple = lambda x, y: np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)

# Generating gridded data. The complex number defines
# how many steps the grid data should have. Without the
# complex number mgrid would only create a grid data structure
# with 5 steps.
grid_x, grid_y = np.mgrid[0:5:1000j, 0:5:1000j]

# Generating sample that interpolation function will see
xy = np.random.rand(1000, 2)

sample = ripple(xy[:,0] * 5 , xy[:,1] * 5)

# Interpolating data with a cubic
grid_z0 = griddata(xy * 5, sample, (grid_x, grid_y), method='cubic')