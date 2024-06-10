import os
from skimage import color
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


# Function to cluster colors in an image using k-means
def cluster_colors(img, num_clusters=8):
    pixels = color.rgb2lab(img[..., :3]).reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(
        pixels
    )
    centers = kmeans.cluster_centers_
    return centers


# display images at the MDS coordinates
def getImage(img):
    return OffsetImage(img, zoom=.5, alpha=1)


images = []

print("Reading images...")
for i in os.listdir("images"):
    img = Image.open(f"images/{i}")
    img = np.array(img)
    images.append(img)
print(f"Read {len(images)} images, each of size {images[0].shape}.")

print("Computing EMD between color distributions...")
# Summarize color distribution of each image
summaries = [cluster_colors(image) for image in images]

# Compute earth mover's distances (EMD) between every pair of signatures
num_images = len(images)
emd_matrix = np.zeros((num_images, num_images))

for i in range(num_images):
    for j in range(i + 1, num_images):
        emd = wasserstein_distance(
            summaries[i].ravel(), summaries[j].ravel()
        )
        emd_matrix[i, j] = emd
        emd_matrix[j, i] = emd

print("Computing MDS embedding...")
# Compute the MDS embedding
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=0,
)
embedding = mds.fit_transform(emd_matrix)

# Plot the MDS embedding
fig, ax = plt.subplots(figsize=(15, 15), dpi=120)
ax.scatter(embedding[:, 0], embedding[:, 1])

for img, coord in zip(images, embedding):
    ab = AnnotationBbox(getImage(img), coord, frameon=False)
    ax.add_artist(ab)

plt.savefig('EMD_colour_dist.png', dpi=1200, bbox_inches="tight")
print("Plot saved as EMD_colour_dist.png")
