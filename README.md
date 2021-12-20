# Clustering algorithms from scratch in Python

> Implementing the [k-Means](https://en.wikipedia.org/wiki/K-means_clustering) and [k-Medoids](https://en.wikipedia.org/wiki/K-medoids) clustering algorithms from scratch in Python.

Clustering algorithms are an unsupervised learning technique used to find patterns in datasets. Our dataset here (in ```MergedData.data```) is a set of word embeddings, i.e. words represented by real-valued vectors such that vectors which are closer in space are closer in meaning. Given a sucessful clustering algorithm, words with similar meanings will be sorted into the same clusters.

```ClusteringAlgorithms.py``` contains main and helper functions implementing the k-Means and k-Medoids 
clustering algorithms plus various examples of its implementation in scripts which generate various graphical results.

Evaluation is performed by the B-CUBED metric, which determines the precision, accuracy, and recall for the produced clustering model.

The ```ClusteringAlgorithms.py``` file implements:

* a method for parsing the data file from text form into a suitable numpy array. If ```normalise=True```, 
all vectors in dataset are normalised to unit lengths.
* a method for implementing the k-Means clustering algorithm to produce k sets of clusters from given 
data parameters. k and r value may be chosen by the user in the scripts
* a method for calculating the objective function for the k-Medoids algorithm
* a method for implementing the k-Medoids clustering algorithm to produce k sets of clusters from given 
data parameters k and r value may be chosen by the user in the scripts
* a method for implementing the B-CUBED algorithm to obtain average precision, accuracy, and f-scores for 
the output of a given clustering algorithm
* a method for plotting the evaluation measures from (5) to visually evaluate the algorithm outputs and 
for evaluating changes to user altered parameters


NOTE: RUNNING SCRIPTS

* Once ran, the scripts will produce a plot for each of the tasks 3 - 6 from the assignment - each plot must be closed before the scripts continue to run.
* ```MergedData.data``` must be in the same folder as the scripts in order for the scripts to execute successfully.
