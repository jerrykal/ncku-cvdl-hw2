import cv2
from sklearn.decomposition import PCA


def dimension_reduction(image):
    """Use PCA to reduce the dimension of the image, then reconstruct the image"""
    if image is None:
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_normalized = gray.astype("float32") / 255

    # Find the mininum number of n that can reconstruct the image with MSE < 3
    n_components = 1
    while True:
        pca = PCA(n_components=n_components)
        pca.fit(gray_normalized)

        # Reconstruct the image
        gray_recovered = pca.inverse_transform(pca.transform(gray_normalized))
        gray_recovered = (gray_recovered * 255).astype("uint8")

        # Calculate MSE
        error = mse(gray, gray_recovered)
        print(f"n_components = {n_components}, MSE = {error}")

        # Stop if MSE < 3
        if error < 3.0:
            cv2.imshow("Recovered Image", gray_recovered)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

        n_components += 1


def mse(original, recovered):
    """Calculate the mean squared error between two images"""
    return ((original - recovered) ** 2).mean()
