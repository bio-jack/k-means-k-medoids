# Jack Brown
# 201056294
# Data Mining and Visualisation - CA2

import numpy as np
import matplotlib.pyplot as plt


def process_data(filename, normalize=False):
    """
    Converts word embeddings in text file into a numpy array.

    Parameters
        filename (string) - name of file to be parsed
        normalize (bool) - apply L2 normalization to vectors representing word embeddings
    Returns
        vectors (np.ndarray) - 2D array, rows are vectors representing word embeddings

    """
    print("Processing dataset...")

    file = open(filename).readlines()
    data = [line.rstrip("\n").split(" ") for line in file]

    # Convert strings to floats
    vectors = []
    for line in data:
        vectors.append([float(v) for v in line[1:]])

    # Convert vectors to numpy arrays
    vectors = np.array([np.array(x) for x in vectors])

    # Apply L2 normalization if normalize=True
    if normalize:
        vectors = vectors / np.linalg.norm(vectors, axis=-1)[:, np.newaxis]

    print("Processing complete.")

    return vectors


def k_means_cluster(vectors, k):

    """
    Clusters a set of vectors into k clusters using the k-Means clustering algorithm.

    Parameters
        vectors (np.ndarray) - 2D array, rows are vectors representing word embeddings
        k       (int) - number of clusters
    Returns
        clusters (list) - list of lists where each list corresponds to a cluster containing indexes
            for objects in that cluster

    """
    # Choose random initial representatives
    rng = np.random.default_rng()
    reprs = rng.choice(len(vectors), size=k, replace=False)

    # Get vectors for initial representatives
    reprs_vectors = [vectors[r] for r in reprs]

    # Iterate over number of epochs, or until convergence
    for iter in range(20):

        # Initialise cluster means
        c_means = []

        # Find the closest representative for each object
        objects_reprs = []  # List of form [[index , vector of closest rep] , [[index , vector of closest rep]...
        for i, object in enumerate(vectors):
            distances = []
            for j, r in enumerate(reprs_vectors):
                distance = np.linalg.norm(r - vectors[i])           # Euclidean distance
                distances.append([r, distance])
            closest_rep = min(distances, key=lambda x: x[1])[0]     # Find index of smallest distance
            objects_reprs.append([i, closest_rep])                  # Append [obectID, vector of closest rep] to objects_reprs

        # Build clusters
        clusters = []
        for r in reprs_vectors:
            clusters.append([])                     # Make a cluster for each vector
            for o_r in objects_reprs:               # Iterate over [objectID, vector of closest rep][...]
                if np.array_equal(o_r[1], r):       # If vector of closest rep == representative
                    clusters[-1].append(o_r[0])     # append objectID to that representative's cluster

        # Compute cluster means
        clusters_as_vectors = [vectors[i] for i in clusters]
        for c in clusters_as_vectors:
            mean = np.mean(c, axis=0)
            c_means.append(mean)

        # Set means as representative vectors for next iteration
        reprs_vectors = c_means

    return clusters


def calculate_obj(objects_reprs):
    """
    Calculates the objective function for a given set of clusters. The objective function is the sum of distances
    between all objects and their representatives.

    Parameters
        objects_reprs (list) - list of lists of the form [[object index, representative]...]
    Returns
        total_distance (float) - the total distance between all objects and their representatives

    """
    total_distance = 0
    for o_r in objects_reprs:
        o, r = o_r
        distance = np.linalg.norm(vectors[o] - r)
        total_distance += distance
    return total_distance


def k_medoids_cluster(vectors, k, r_val):
    """
    Clusters a set of vectors into k clusters using the k-Medoids clustering algorithm.

    Parameters
        vectors (np.ndarray) - 2D array, rows are vectors representing word embeddings
        k (int) - number of clusters
        r_val (int) - number of objects to sample to replace representatives at each iteration
    Returns
        clusters (list) - list of lists where each list corresponds to a cluster containing indexes
            for objects in that cluster

    """
    # Choose random initial representatives
    rng = np.random.default_rng()
    reprs = rng.choice(len(vectors), size=k, replace=False)

    # Get vectors for initial representatives
    reprs_vectors = [vectors[r] for r in reprs]

    # For building final clusters
    final_objects_reprs = []

    # Iterate over number of epochs, or until convergence
    for iter in range(20):

        # Find the closest representative for each object
        objects_reprs = []  # List of form [[index , vector of closest rep] , [[index , vector of closest rep]...
        for i, object in enumerate(vectors):
            distances = []
            for j, r in enumerate(reprs_vectors):
                distance = np.linalg.norm(r - object)           # Euclidean distance
                distances.append([r, distance])
            closest_rep = min(distances, key=lambda x: x[1])[0]     # Find index of smallest distance
            objects_reprs.append([i, closest_rep])                  # Append [obectID, vector of closest rep] to objects_reprs

        # Compute obj function - sum of distances of objects from their repr
        prev_total_distance = calculate_obj(objects_reprs)

        # Randomly select r pairs of X (objects in data) and Y (representatives) for swapping
        X_choices = rng.choice(len(vectors), size=r_val, replace=False)
        X = [vectors[x] for x in X_choices]
        Y_indeces = rng.choice(len(reprs_vectors), size=r_val)

        X_Y = zip(X, Y_indeces)

        # Initialise list of obj function values for possible replacement representatives
        new_total_distances = []

        # Loop over each pair of X, Y swaps:
        for x, yi in X_Y:
            # Temporary list of representative vectors
            temp_repr_vector = reprs_vectors

            # Replace current values of Y and X in temporary representatives
            temp_repr_vector[yi] = x

            # Re-assign all objects
            temp_objects_reprs = []  # List of form [[index , vector of closest rep]...]
            for i, object in enumerate(vectors):
                distances = []
                for j, r in enumerate(temp_repr_vector):
                    distance = np.linalg.norm(r - vectors[i])
                    distances.append([r, distance])
                closest_rep = min(distances, key=lambda x: x[1])[0]
                temp_objects_reprs.append([i, closest_rep])

            # Re-compute objective function
            new_total_distance = calculate_obj(temp_objects_reprs)
            new_total_distances.append(new_total_distance)

        # Check for convergence - if optimisation step does not improve objective function - terminate
        min_total_distance = np.min(new_total_distances)
        # Else continue with iteration
        if min_total_distance >= prev_total_distance:
            break

        # Find which pair swap gives largest decrease in objective function
        min_total_distance_index = np.argmin(new_total_distances)
        best_X = X_choices[min_total_distance_index]
        Y_to_swap = Y_indeces[min_total_distance_index]

        # Update representatives for next loop
        reprs_vectors[Y_to_swap] = vectors[best_X]

    # Re-compute objects_reprs based on final reprs_vectors
    objects_reprs = []  # List of form [[index , vector of closest rep] , [[index , vector of closest rep]...
    for i, object in enumerate(vectors):
        distances = []
        for j, r in enumerate(reprs_vectors):
            distance = np.linalg.norm(r - vectors[i])  # Euclidean distance
            distances.append([r, distance])
        closest_rep = min(distances, key=lambda x: x[1])[0]  # Find index of smallest distance
        objects_reprs.append([i, closest_rep])  # Append [obectID, vector of closest rep] to objects_reprs

    # Build final clusters
    clusters = []
    for r in reprs_vectors:
        clusters.append([])  # Make a cluster for each vector
        for o_r in objects_reprs:
            if np.array_equal(o_r[1], r):
                clusters[-1].append(o_r[0])

    return clusters


def BCUBED(clusters, labels):
    """
    Evaluates clustering using the BCUBED algorithm.

    Parameters
        clusters (list) - list of lists where each list corresponds to a cluster containing indexes
            for objects in that cluster
        labels (dict) - dict of the form {key=objectID : label=0(animal)/1(country)/...}
    Returns
        avg_precision (float) - average precision over all objects in the dataset
        avg_recall (float) - average recall over all objects in the dataset
        avg_f_score (float) - average f-score over all objects in the dataset

    """
    # Convert clusters of objects into labels, e.g. [[0, 0 , 0, 1], [1, 1, 1, 1, 2]...]
    clusters_as_labels = [[labels[obj] for obj in c] for c in clusters]

    # Calculate precision and recall scores for each object
    precision_scores = []
    recall_scores = []
    for i, cluster in enumerate(clusters):
        for obj in cluster:
            label = labels[obj]
            label_count_in_cluster = clusters_as_labels[i].count(label)
            cluster_size = len(cluster)
            label_count_in_dataset = sum(cluster_labels.count(label) for cluster_labels in clusters_as_labels)

            precision = label_count_in_cluster / cluster_size
            precision_scores.append(precision)

            recall = label_count_in_cluster / label_count_in_dataset
            recall_scores.append(recall)

    # Calculate f-scores for each object
    f_scores = []
    for p, r in zip(precision_scores, recall_scores):
        f_score = (2 * p * r) / (p + r)
        f_scores.append(f_score)

    # Calculate averages across dataset
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f_score = np.mean(f_scores)

    return avg_precision, avg_recall, avg_f_score


def plot_BCUBED(k, avg_precision, avg_recall, avg_f_score):
    """
    Plots outputs of the B-CUBED algorithm against the number of clusters.

    Parameters
        k (int) - number of clusters
        avg_precision (float) - average precision over all objects in the dataset
        avg_recall (float) - average recall over all objects in the dataset
        avg_f_score (float) - average f-score over all objects in the dataset
    Returns
        None

    """
    x = range(1, k+1)

    plt.plot(x, avg_precision, label="Avg precision")
    plt.plot(x, avg_recall, label="Avg recall")
    plt.plot(x, avg_f_score, label="Avg F-scores")

    plt.xlabel = "k"
    plt.ylabel = "Scores"
    plt.title("Evaluating algorithm at different values of k.")

    plt.legend()
    plt.show()


if __name__ == "__main__":

    # Create labels dict for computing BCUBED score {key=objectID : label=0(animal)/1(country)/...}
    labels = {}
    for i in range(50):
        labels[i] = 0
    for i in range(50, 211):
        labels[i] = 1
    for i in range(211, 269):
        labels[i] = 2
    for i in range(269, 329):
        labels[i] = 3

    # ###
    # # Task 3 - Plot evaluation scores vs k = [1, 2,..., k] for k-Means algorithm
    # ###

    print("Plotting evaluation scores for k-Means algorithm...")

    vectors = process_data("MergedData.txt")

    avg_precisions = []
    avg_recalls = []
    avg_f_scores = []

    for k in range(1, 10):
        clusters = k_means_cluster(vectors, k)
        avg_precision, avg_recall, avg_f_score = BCUBED(clusters, labels)
        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)
        avg_f_scores.append(avg_f_score)

    print("Close plot to continue.")
    plot_BCUBED(9, avg_precisions, avg_recalls, avg_f_scores)


    # ###
    # # Task 4 - Same as above, but with normalized object vectors
    # ###

    print("\nPlotting evaluation scores for k-Means algorithm w/ normalization...")
    vectors = process_data("MergedData.txt", normalize=True)
    avg_precisions = []
    avg_recalls = []
    avg_f_scores = []

    for k in range(1, 10):
        clusters = k_means_cluster(vectors, k)
        avg_precision, avg_recall, avg_f_score = BCUBED(clusters, labels)
        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)
        avg_f_scores.append(avg_f_score)

    print("Close plot to continue.")
    plot_BCUBED(9, avg_precisions, avg_recalls, avg_f_scores)


    ###
    # Task 5 - Plot evaluation scores vs k = [1, 2,..., k] for k-Medoids algorithm
    ###

    print("\nPlotting evaluation scores for k-Medoids algorithm...")
    vectors = process_data("MergedData.txt", normalize=False)
    avg_precisions = []
    avg_recalls = []
    avg_f_scores = []

    for k in range(1, 10):
        clusters = k_medoids_cluster(vectors, k, r_val=10)
        avg_precision, avg_recall, avg_f_score = BCUBED(clusters, labels)
        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)
        avg_f_scores.append(avg_f_score)
    print("Close plot to continue.")
    plot_BCUBED(9, avg_precisions, avg_recalls, avg_f_scores)


    # ###
    # # Task 6 - Same as above, but with normalized vectors
    # ###

    print("\nPlotting evaluation scores for k-Medoids algorithm w/ normalization...")
    vectors = process_data("MergedData.txt", normalize=True)
    avg_precisions = []
    avg_recalls = []
    avg_f_scores = []

    for k in range(1, 10):
        clusters = k_medoids_cluster(vectors, k, r_val=10)
        avg_precision, avg_recall, avg_f_score = BCUBED(clusters, labels)
        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)
        avg_f_scores.append(avg_f_score)

    print("Close plot to continue.")
    plot_BCUBED(9, avg_precisions, avg_recalls, avg_f_scores)
