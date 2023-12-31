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

# samples = [0.23994708, 0.90414377, 0.0, 1.0, 0.09654337, 0.0, 0.94947183, 1.0, 0.34800794, 0.0, 0.09915567, 1.0, 0.10923902, 0.30830924, 1.0, 0.01785592, 0.31967051, 0.9693579, 0.2111661, 0.08140928, 0.25608037, 0.29938609, 0.09927996, 0.40257879, 0.37492004, 0.34055429, 0.0, 0.77826134, 0.07747039, 0.20981656, 1.0, 1.0, 1.0, 0.05086198, 0.30375049, 0.90540764, 1.0, 1.0, 0.94857492, 0.9746724, 0.93737285, 0.42447949, 0.37552089, 0.15774268, 0.3215633, 0.3970699, 0.1334957, 0.40928235, 0.91639812, 0.14531311, 0.18193168, 0.0103724, 0.3072668, 0.06571542, 0.40856247, 0.0, 0.89891865, 0.83690982]

# actual FST samples
samples = [0.31776233823647615, 0.3183453237410072, 0.2956208856971873, 0.3029761904761905, 0.15900288557541445, 0.07322175732217573, 0.9991974317817014, 0.21980196543898156, 0.0010460251046025104, 0.8657640015666089, 0.2013467475779836, 0.40952759482171247, 1.0, 1.0, 0.9755289391943536, 1.0, 0.1403345724907063, 0.9825, 0.0028957528957528956, 0.2378571029239112, 0.0, 0.0, 1.0, 0.38033133143322473, 0.8106720430107527, 0.9581320450885669, 0.06891527875313132, 0.10814388395615832, 0.36808243243243244, 0.9992260061919506, 0.12797074954296161, 0.04291044776119403, 0.005681818181818182, 0.5616122464959674, 0.9992012779552717, 0.1808599419448476, 0.39473580811103254, 0.015677060765886236, 0.7186034608017964, 0.5982344215439164, 0.4574360016253555, 0.9788059173353727, 0.26131034938247144, 0.19669325321455067, 0.8695857516786161, 0.06836734693877551, 0.5261361200428725, 0.36939641683617586, 0.30381858285084096, 0.4816518156558398, 0.24008795197542931, 0.332939065176605, 1.0, 1.0, 0.1487845923709798, 0.3283452016292164, 0.6547331495865698, 1.0]

# Reshape data for Gaussian Mixture Model fitting
data_2d = np.array(samples).reshape(-1, 1)

# Fit bimodal distribution
#gmm_bimodal = GaussianMixture(n_components=2, random_state=0, means_init=np.array([0.25, 1]).reshape(-1,1)).fit(data_2d)
gmm_bimodal = GaussianMixture(n_components=2, random_state=0, init_params='k-means++', n_init=1000).fit(data_2d)


# Fit trimodal distribution
#gmm_trimodal = GaussianMixture(n_components=3, random_state=0, means_init=np.array([0, 0.33, 1]).reshape(-1,1)).fit(data_2d)
gmm_trimodal = GaussianMixture(n_components=3, random_state=0, init_params='k-means++', n_init=1000).fit(data_2d)

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
plt.hist(samples, bins=25, density=True, alpha=0.6, color='gray')

# PDF of the bimodal fit
plt.plot(x, gmm_density(gmm_bimodal, x), label='Bimodal Fit', color='blue')

# PDF of the trimodal fit
plt.plot(x, gmm_density(gmm_trimodal, x), label='Trimodal Fit', color='red')

plt.title('Data with Bimodal and Trimodal Gaussian Mixture Model Fits')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.show()
