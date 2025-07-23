import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Empirical function to compute the CDF
def empirical_CDF_function(sample, y_values):
    n = len(sample)
    sorted_sample = np.sort(sample)
    cdf_values = np.array([np.sum(sorted_sample <= y) / n for y in y_values])
    return cdf_values

# Compute the survival function from CDF
def empirical_survival_function_from_cdf(cdf_values):
    return 1 - cdf_values

# Compute \xi_{empirical}
def compute_xi_empirical(sample, y_values):
    cdf_values = empirical_CDF_function(sample, y_values)
    survival_values = empirical_survival_function_from_cdf(cdf_values)
    survival_values = np.maximum(survival_values, 1e-10)  # Avoid log(0)
    xi_empirical = -np.sum(survival_values * np.log(survival_values) * (y_values[1] - y_values[0]))
    return xi_empirical

# Compute JCRE
def compute_JCRE(image1, image2, y_values, p):
    # Flatten images into 1D arrays (pixels)
    sample1 = image1.flatten()
    sample2 = image2.flatten()

    # Compute \xi(\bar{F}_1), \xi(\bar{F}_2), and \xi(p\bar{F}_1 + (1-p)\bar{F}_2)
    cdf_values1 = empirical_CDF_function(sample1, y_values)
    cdf_values2 = empirical_CDF_function(sample2, y_values)

    survival_values1 = empirical_survival_function_from_cdf(cdf_values1)
    survival_values2 = empirical_survival_function_from_cdf(cdf_values2)

    # Compute the convex combination p\bar{F}_1 + (1-p)\bar{F}_2
    combined_survival = p * survival_values1 + (1 - p) * survival_values2
    combined_survival = np.maximum(combined_survival, 1e-10)  # Avoid log(0)

    # Compute empirical estimate of \xi for the combination
    xi_combined = -np.sum(combined_survival * np.log(combined_survival) * (y_values[1] - y_values[0]))

    # Compute empirical estimate of \xi for each image
    xi1 = compute_xi_empirical(sample1, y_values)
    xi2 = compute_xi_empirical(sample2, y_values)

    # Compute empirical estimate of JCRE
    jcre = xi_combined - p * xi1 - (1 - p) * xi2

    return jcre

# Function to add Gaussian noise to an image
def add_noise(image, snr_db):
    """Add Gaussian noise to an image based on the desired SNR in dB."""
    snr = 10 ** (snr_db / 10.0)
    signal_power = np.mean(image ** 2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape).astype(np.float32)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)

# Function to convert a colored image to grayscale
def to_gray(image):
    """Convert a colored image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load a new image
image_path = "C:/Users/FARHANG/Desktop/image/Road_in_Norway.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Please provide a valid path to the image.")

# Original and grayscale images
colored_image = image
gray_image = to_gray(image)

# Resize grayscale image for processing
image_resized = cv2.resize(gray_image, (256, 256)).astype(np.float32)

# Define SNR values in dB
snr_values = [40, 20, 10, 0]
noisy_images = [add_noise(image_resized, snr) for snr in snr_values]

# y-values at which the CDF is computed
y_values = np.linspace(0, 255, 256)

# p-values for convex combination
p_values = [0.2, 0.5, 0.8]

# Compute JCRE for noisy images at different SNRs and p-values
JCRE_values = {}
for p in p_values:
    JCRE_values[p] = []
    for snr, noisy_image in zip(snr_values, noisy_images):
        jcre_value = compute_JCRE(image_resized, noisy_image, y_values, p)
        JCRE_values[p].append(jcre_value)
        print(f"Empirical JCRE for p={p} and SNR={snr} dB: {jcre_value:.4f}")

# Plot noisy images and intensity profiles
fig, axes = plt.subplots(2, len(snr_values), figsize=(10, 6))  # Adjusted figure size

for i, (snr, noisy_image) in enumerate(zip(snr_values, noisy_images)):
    # Display noisy image
    axes[0, i].imshow(noisy_image, cmap='gray')
    jcre_value = JCRE_values[0.5][i]  # Use JCRE value for p=0.5
    axes[0, i].set_title(f"SNR: {snr} dB\n$JCRE(\\bar{{F}},\\bar{{G}}) = {jcre_value:.4f}$")
    axes[0, i].axis('off')

    # Extract intensity profile for a selected region (example: entire image)
    mean_profile = np.mean(noisy_image, axis=0)
    x_axis = np.arange(len(mean_profile))

    # Plot intensity profile
    axes[1, i].plot(x_axis, mean_profile, color='blue', label='Noisy')
    axes[1, i].plot(x_axis, np.mean(image_resized, axis=0), color='red', label='Original')
    axes[1, i].set_title(f"SNR = {snr} dB")
    axes[1, i].legend()

plt.tight_layout()
plt.show()

# Plot histograms for noisy images
fig_hist, axes_hist = plt.subplots(1, len(snr_values), figsize=(10, 4))  # Adjusted figure size

for i, (snr, noisy_image) in enumerate(zip(snr_values, noisy_images)):
    # Plot histogram of noisy image with blue bars and red KDE curve
    axes_hist[i].hist(noisy_image.flatten(), bins=50, color='blue', alpha=0.7, density=True)
    sns.kdeplot(noisy_image.flatten(), color='red', ax=axes_hist[i], lw=2)
    axes_hist[i].set_title(f"Histogram (SNR = {snr} dB)")
    axes_hist[i].set_xlabel('Pixel Intensity')
    axes_hist[i].set_ylabel('Density')

plt.tight_layout()
plt.show()
