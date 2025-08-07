
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












###########################################################################################
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

# Read the image
image_path = "C:/Users/....."
try:
    im = imread(image_path)
    print("Image successfully loaded.")
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}.")
    exit(1)

# Ensure the image is in grayscale if it has multiple channels
if im.ndim == 3:  # If the image is RGB
    im = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
    print("Converted RGB image to grayscale.")

# Display the original grayscale image
plt.figure()
plt.imshow(im, cmap=plt.cm.gray)
plt.axis('off')
plt.title("Grayscale Image")
plt.show()

# Display the histogram
plt.figure()
hist, bins = np.histogram(im.flatten(), bins=np.arange(257), density=True)
plt.hist(im.flatten(), bins=np.arange(257), density=True, color='blue', alpha=0.7)
plt.title("Histogram of Pixel Intensities")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

# Function to compute JCRE for a histogram
def compute_JCRE_histogram(h, t, w):
    term_3 = 0
    for s in range(1, len(p1) + 1):
        for r in range(1, len(p2) + 1):
            value = (w *(np.sum(p1[:s])) + (1 - w) *(np.sum(p2[:r]))) * np.log(w *(np.sum(p1[:s])) + (1 - w) *(np.sum(p2[:r]))+ee)
            term_3 += value

    return term_1 + term_2 - term_3

# JCRE-based thresholding function
def jcre_threshold(h):
    ee =1e-10
    h = h / h.sum()
    # Normalize histogram
    total_w = np.sum([np.sum(h[:t]) for t in range(1, len(h))])
    jcre_values = np.zeros(len(h))

    for t in range(1, len(h) - 1):
        w = np.sum(h[:t])
        jcre_values[t] = compute_JCRE_histogram(h, t, w)

    best_t = np.argmax(jcre_values)/k
    return best_t, jcre_values

# Compute the JCRE threshold
h, bins = np.histogram(im.flatten(), bins=np.arange(257))
threshold, jcre_values = jcre_threshold(h)
print(f"JCRE threshold: {threshold}")



# Apply Otsu's thresholding method to determine the optimal threshold
otsu_threshold = threshold_otsu(im)
print(f"Otsu's threshold: {otsu_threshold}")

# Apply the threshold to segment the image using Otsu's method
segmented_image_otsu = (im > otsu_threshold).astype(int)

# K-means clustering for image segmentation
# Reshape the image into a 2D array (pixels, features)
pixels = im.flatten().reshape(-1, 1)

# Apply K-means with 2 clusters (binary segmentation)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(pixels)
segmented_image_kmeans = kmeans.labels_.reshape(im.shape)

# Reverse the K-means labels (invert segmentation)
kmeans_cluster_centers = kmeans.cluster_centers_.flatten()
if kmeans_cluster_centers[0] > kmeans_cluster_centers[1]:
    segmented_image_kmeans = 1 - segmented_image_kmeans


# Compute evaluation metrics
def compute_metrics(ground_truth, segmented_image):
    ground_truth_flat = ground_truth.flatten()
    segmented_image_flat = segmented_image.flatten()

    TP = np.sum((segmented_image_flat == 1) & (ground_truth_flat == 1))
    FP = np.sum((segmented_image_flat == 1) & (ground_truth_flat == 0))
    TN = np.sum((segmented_image_flat == 0) & (ground_truth_flat == 0))
    FN = np.sum((segmented_image_flat == 0) & (ground_truth_flat == 1))

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    dice_coefficient = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    jaccard_index = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    ari_value = adjusted_rand_score(ground_truth_flat, segmented_image_flat)

    return accuracy, recall, precision, specificity, f1_score, dice_coefficient, jaccard_index, ari_value

# Calculate metrics for JCRE thresholding
metrics_jcre = compute_metrics(ground_truth, segmented_image_jcre)
metrics_otsu = compute_metrics(ground_truth, segmented_image_otsu)
metrics_kmeans = compute_metrics(ground_truth, segmented_image_kmeans)

metric_names = ["Accuracy", "Recall", "Precision", "Specificity", "F1-Score", "Dice Coefficient", "Jaccard Index", "Adjusted Rand Index (ARI)"]

print("\nMetrics for JCRE Thresholding:")
for name, value in zip(metric_names, metrics_jcre):
    print(f"{name}: {value}")

print("\nMetrics for Otsu Thresholding:")
for name, value in zip(metric_names, metrics_otsu):
    print(f"{name}: {value}")

print("\nMetrics for K-means Segmentation:")
for name, value in zip(metric_names, metrics_kmeans):
    print(f"{name}: {value}")



# Display Segmentation Results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Ground Truth
axes[0, 0].imshow(ground_truth, cmap=plt.cm.gray)
axes[0, 0].axis('off')
axes[0, 0].set_title("Ground Truth (GT)")

# Jensen-GS Segmented Image
axes[0, 1].imshow(segmented_image_jcre, cmap=plt.cm.gray)
axes[0, 1].axis('off')
axes[0, 1].set_title("Segmented Image ( Algorithm $I_{S}^{*}$)")

# Otsu Segmented Image
axes[1, 0].imshow(segmented_image_otsu, cmap=plt.cm.gray)
axes[1, 0].axis('off')
axes[1, 0].set_title("Segmented Image (Otsu)")

# K-means Segmented Image
axes[1, 1].imshow(segmented_image_kmeans, cmap=plt.cm.gray)
axes[1, 1].axis('off')
axes[1, 1].set_title("Segmented Image (K-means)")

plt.tight_layout()
# Save plot
save_path = "C:/Users/...."
plt.savefig(save_path, format='png', dpi=300)
print(f"Plot saved successfully at {save_path}")
plt.show()

# Display Original, Grayscale, and Histogram in One Window
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original Colored Image
original_image = imread(image_path)
axes[0].imshow(original_image)
axes[0].axis('off')
axes[0].set_title("Original Colored Image")

# Grayscale Image
axes[1].imshow(im, cmap=plt.cm.gray)
axes[1].axis('off')
axes[1].set_title("Grayscale Image")

# Histogram of Grayscale Image
hist, bins = np.histogram(im.flatten(), bins=np.arange(257), density=True)
axes[2].hist(im.flatten(), bins=np.arange(257), density=True, color='blue', alpha=0.7)
axes[2].set_title("Histogram of Pixel Intensities")
axes[2].set_xlabel("Pixel Value")
axes[2].set_ylabel("Frequency")
plt.tight_layout()

# Save the plot
save_path_tt1 = "C:/Users/...."
plt.savefig(save_path_tt1, format='png', dpi=300)
print(f"Combined plot saved successfully at {save_path_tt1}")
plt.show()

