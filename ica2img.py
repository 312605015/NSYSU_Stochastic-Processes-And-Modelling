from PIL import Image
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Load the images
image1 = Image.open("image1.jpg").convert("L")  # Convert to grayscale
image2 = Image.open("image2.jpg").convert("L")
image3 = Image.open("image3.jpg").convert("L")

# Convert images to numpy arrays
image1_array = np.array(image1)
image2_array = np.array(image2)
image3_array = np.array(image3)

# Normalize the image arrays
image1_normalized = image1_array / 255.0
image2_normalized = image2_array / 255.0
image3_normalized = image3_array / 255.0

# Reshape the image arrays to 1D
image1_flattened = image1_normalized.reshape(-1)
image2_flattened = image2_normalized.reshape(-1)
image3_flattened = image3_normalized.reshape(-1)

# Perform image mixing
mixing_matrix = np.array([[1, 0.5, 1.5], [1, 2, 1], [1.5, 1, 2]])
mixed_images = np.dot(mixing_matrix, np.array([image1_flattened, image2_flattened, image3_flattened]))

# Reshape the mixed images back to 2D
mixed_image1 = mixed_images[0].reshape(image1_array.shape)
mixed_image2 = mixed_images[1].reshape(image2_array.shape)
mixed_image3 = mixed_images[2].reshape(image3_array.shape)

# Perform Blind Source Separation using FastICA
ica = FastICA(n_components=3)
recovered_images = ica.fit_transform(mixed_images.T).T

# Reshape the recovered images back to 2D
recovered_image1 = recovered_images[0].reshape(image1_array.shape)
recovered_image2 = recovered_images[1].reshape(image2_array.shape)
recovered_image3 = recovered_images[2].reshape(image3_array.shape)

# Display the original images
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(image1_array, cmap="gray")
plt.title("Original Image 1")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(image2_array, cmap="gray")
plt.title("Original Image 2")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(image3_array, cmap="gray")
plt.title("Original Image 3")
plt.axis("off")

# Display the mixed images
plt.subplot(2, 3, 4)
plt.imshow(mixed_image1, cmap="gray")
plt.title("Mixed Image 1")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(mixed_image2, cmap="gray")
plt.title("Mixed Image 2")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(mixed_image3, cmap="gray")
plt.title("Mixed Image 3")
plt.axis("off")

plt.tight_layout()
plt.show()

# Display the recovered images
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(recovered_image1, cmap="gray")
plt.title("Recovered Image 1")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(recovered_image2, cmap="gray")
plt.title("Recovered Image 2")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(recovered_image3, cmap="gray")
plt.title("Recovered Image 3")
plt.axis("off")

plt.tight_layout()
plt.show()
