import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def dimension_reduction(image):
    """Use PCA to reduce the dimension of the image, then reconstruct the image"""
    if image is None:
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_normalized = gray.astype("float32") / 255

    # Find the mininum number of n that can reconstruct the image with MSE < 3
    n_components = 80
    gray_recovered = None
    while n_components < min(gray.shape):
        pca = PCA(n_components=n_components)

        # Reconstruct the image
        gray_recovered = pca.inverse_transform(pca.fit_transform(gray_normalized))

        # Calculate MSE
        error = mse(gray, (gray_recovered * 255).astype("uint8"))
        print(f"n_components = {n_components}, MSE = {error}")

        # Stop if MSE < 3
        if error < 3.0:
            break

        n_components += 1

    if gray_recovered is None:
        return

    # Plot the original image, the gray scale image and the reconstructed image
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].imshow(gray_normalized, cmap="gray")
    axes[1].set_title("Gray Scale Image")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[2].imshow(gray_recovered, cmap="gray")
    axes[2].set_title(f"Reconstructed Image with {n_components} Components")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.show()


def mse(original, recovered):
    """Calculate the mean squared error between two images"""
    return ((original - recovered) ** 2).mean()
