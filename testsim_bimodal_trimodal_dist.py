import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Parameters for the tri-modal distribution
means = [0, 0.33, 1]  # Means of the three modes
std_dev = 0.1  # Standard deviation for more overlap
samples_per_mode = [19, 19, 20]  # Splitting 58 samples approximately equally

# Generating the samples
'''
samples = []
for mean, num in zip(means, samples_per_mode):
    samples.extend(np.random.normal(mean, std_dev, num))
samples = np.clip(samples, 0, 1)  # Ensure values are within [0, 1]
'''

samples = [0.23994708, 0.90414377, 0.0, 1.0, 0.09654337, 0.0, 0.94947183, 1.0, 0.34800794, 0.0, 0.09915567, 1.0, 0.10923902, 0.30830924, 1.0, 0.01785592, 0.31967051, 0.9693579, 0.2111661, 0.08140928, 0.25608037, 0.29938609, 0.09927996, 0.40257879, 0.37492004, 0.34055429, 0.0, 0.77826134, 0.07747039, 0.20981656, 1.0, 1.0, 1.0, 0.05086198, 0.30375049, 0.90540764, 1.0, 1.0, 0.94857492, 0.9746724, 0.93737285, 0.42447949, 0.37552089, 0.15774268, 0.3215633, 0.3970699, 0.1334957, 0.40928235, 0.91639812, 0.14531311, 0.18193168, 0.0103724, 0.3072668, 0.06571542, 0.40856247, 0.0, 0.89891865, 0.83690982]

# Reshape data for Gaussian Mixture Model fitting
data_2d = np.array(samples).reshape(-1, 1)

# Fit bimodal distribution
gmm_bimodal = GaussianMixture(n_components=2, random_state=0).fit(data_2d)

# Fit trimodal distribution
gmm_trimodal = GaussianMixture(n_components=3, random_state=0).fit(data_2d)

# Calculate AIC and BIC for both models
aic_bimodal = gmm_bimodal.aic(data_2d)
bic_bimodal = gmm_bimodal.bic(data_2d)
aic_trimodal = gmm_trimodal.aic(data_2d)
bic_trimodal = gmm_trimodal.bic(data_2d)

# Output the results
print("Bimodal Fit - AIC:", aic_bimodal, "BIC:", bic_bimodal)
print("Trimodal Fit - AIC:", aic_trimodal, "BIC:", bic_trimodal)


# Function to calculate the density for GMM
def gmm_density(gmm, x):
    return np.exp(gmm.score_samples(x.reshape(-1, 1)))

# Create a range of values for plotting the PDF
x = np.linspace(0, 1, 1000)

# Plotting
plt.figure(figsize=(12, 6))

# Histogram of the data
plt.hist(samples, bins=15, density=True, alpha=0.6, color='gray')

# PDF of the bimodal fit
plt.plot(x, gmm_density(gmm_bimodal, x), label='Bimodal Fit', color='blue')

# PDF of the trimodal fit
plt.plot(x, gmm_density(gmm_trimodal, x), label='Trimodal Fit', color='red')

plt.title('Data with Bimodal and Trimodal Gaussian Mixture Model Fits')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.show()
