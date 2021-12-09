=====================
CLUSTERING ALGORITHMS
=====================

"ClusteringAlgorithms.py" contains main and help functions implementing the k-Means and k-Medoids 
clustering algorithms plus various examples of its implementation in scripts which generate the various 
results required by the assignment.

It also contains "MergedData.data" which must be in the same folder as the "ClusteringAlgorithms.py" for
the scripts to run and display the required results correctly.

The ClusteringAlgorithms.py file implements:

(1) a method for parsing the data file from text form into a suitable numpy array form if normalise=True, 
all vectors in dataset are normalised to unit lengths

(2) a method for implementing the k-Means clustering algorithm to produce k sets of clusters from given 
data parameters k and r value may be chosen by the user in the scripts

(3) a method for calculating the objective function for the k-Medoids algorithm

(4) a method for implementing the k-Medoids clustering algorithm to produce k sets of clusters from given 
data parameters k and r value may be chosen by the user in the scripts

(5) a method for implementing the B-CUBED algorithm to obtain average precision, accuracy, and f-scores for 
the output of a given clustering algorithm

(6) a method for plotting the evaluation measures from (5) to visually evaluate the algorithm outputs and 
for evaluating changes to user altered parameters


NOTE: RUNNING SCRIPTS

-- Once ran, the scripts will produce a plot for each of the tasks 3 - 6 from the assignment - each
   plot must be closed before the scripts continue to run.

-- "MergedData.data" must be in the same folder as the scripts in order for the scripts to execute
   successfully.
