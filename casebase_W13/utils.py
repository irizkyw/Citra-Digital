import numpy as np

def region_growing(image, seeds, threshold=5):
    segmented = np.zeros_like(image)
    for seed in seeds:
        x, y = seed
        region_value = image[x, y]
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if segmented[x, y] == 0:
                segmented[x, y] = region_value
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                        if segmented[nx, ny] == 0 and abs(int(image[nx, ny]) - int(region_value)) <= threshold:
                            stack.append((nx, ny))
    return segmented



def split_merge(image, threshold=5):
    def split(image):
        h, w = image.shape
        if h == 1 or w == 1:
            return [image]
        h2, w2 = h // 2, w // 2
        return split(image[:h2, :w2]) + split(image[:h2, w2:]) + split(image[h2:, :w2]) + split(image[h2:, w2:])
    
    def merge(segments):
        h, w = len(segments), len(segments[0])
        merged = np.zeros((h, w), dtype=int)
        for i in range(h):
            for j in range(w):
                merged[i, j] = np.mean(segments[i][j])
        return merged
    
    segments = split(image)
    merged_image = merge(segments)
    return merged_image


def KMeans(image, k=2, max_iter=100):
    def distance(x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    
    def assign_clusters(image, centroids):
        h, w = image.shape
        clusters = np.zeros((h, w), dtype=int)
        for i in range(h):
            for j in range(w):
                distances = [distance(image[i, j], centroid) for centroid in centroids]
                clusters[i, j] = np.argmin(distances)
        return clusters
    
    def update_centroids(image, clusters, k):
        h, w = image.shape
        centroids = np.zeros((k, 3), dtype=float)
        for i in range(k):
            cluster = np.where(clusters == i)
            if len(cluster[0]) == 0:
                continue
            for j in range(3):
                centroids[i, j] = np.mean(image[cluster[0], cluster[1], j])
        return centroids
    
    h, w, _ = image.shape
    image = image.reshape(-1, 3)
    centroids = image[np.random.choice(range(h * w), k, replace=False)]
    clusters = np.zeros(h * w, dtype=int)
    for _ in range(max_iter):
        new_clusters = assign_clusters(image, centroids)
        if np.array_equal(clusters, new_clusters):
            break
        clusters = new_clusters
        centroids = update_centroids(image, clusters, k)
    return clusters.reshape(h, w)