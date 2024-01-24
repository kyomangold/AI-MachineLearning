# Implementation of a decision tree from scratch for the task of classifying whether a mushroom is edible or poisonous.

import numpy as np
import matplotlib.pyplot as plt
from public_tests import *
from utils import *

# Problem Statement

# Suppose you are starting a company that grows and sells wild mushrooms. 
# - Since not all mushrooms are edible, you'd like to be able to tell whether a given mushroom is edible or poisonous based on it's physical attributes
# - You have some existing data that you can use for this task. 
# 
# Can you use the data to help you identify which mushrooms can be sold safely? 
# 
# Note: The dataset used is for illustrative purposes only. It is not meant to be a guide on identifying edible mushrooms.

# Dataset
# 
# The dataset looks as follows:
# 
# |                                                     | Cap Color | Stalk Shape | Solitary | Edible |
# |:---------------------------------------------------:|:---------:|:-----------:|:--------:|:------:|
# | <img src="images/0.png" alt="drawing" width="50"/> |   Brown   |   Tapering  |    Yes   |    1   |
# | <img src="images/1.png" alt="drawing" width="50"/> |   Brown   |  Enlarging  |    Yes   |    1   |
# | <img src="images/2.png" alt="drawing" width="50"/> |   Brown   |  Enlarging  |    No    |    0   |
# | <img src="images/3.png" alt="drawing" width="50"/> |   Brown   |  Enlarging  |    No    |    0   |
# | <img src="images/4.png" alt="drawing" width="50"/> |   Brown   |   Tapering  |    Yes   |    1   |
# | <img src="images/5.png" alt="drawing" width="50"/> |    Red    |   Tapering  |    Yes   |    0   |
# | <img src="images/6.png" alt="drawing" width="50"/> |    Red    |  Enlarging  |    No    |    0   |
# | <img src="images/7.png" alt="drawing" width="50"/> |   Brown   |  Enlarging  |    Yes   |    1   |
# | <img src="images/8.png" alt="drawing" width="50"/> |    Red    |   Tapering  |    No    |    1   |
# | <img src="images/9.png" alt="drawing" width="50"/> |   Brown   |  Enlarging  |    No    |    0   |
# 
# 
# - 10 examples of mushrooms. For each example, 
#     - Three features
#         - Cap Color (`Brown` or `Red`),
#         - Stalk Shape (`Tapering (as in \/)` or `Enlarging (as in /\)`), and
#         - Solitary (`Yes` or `No`)
#     - Label
#         - Edible (`1` indicating yes or `0` indicating poisonous)

# One hot encoded dataset
# For ease of implementation, have one-hot encoded the features (turned them into 0 or 1 valued features)
# 
# |                                                    | Brown Cap | Tapering Stalk Shape | Solitary | Edible |
# |:--------------------------------------------------:|:---------:|:--------------------:|:--------:|:------:|
# | <img src="images/0.png" alt="drawing" width="50"/> |     1     |           1          |     1    |    1   |
# | <img src="images/1.png" alt="drawing" width="50"/> |     1     |           0          |     1    |    1   |
# | <img src="images/2.png" alt="drawing" width="50"/> |     1     |           0          |     0    |    0   |
# | <img src="images/3.png" alt="drawing" width="50"/> |     1     |           0          |     0    |    0   |
# | <img src="images/4.png" alt="drawing" width="50"/> |     1     |           1          |     1    |    1   |
# | <img src="images/5.png" alt="drawing" width="50"/> |     0     |           1          |     1    |    0   |
# | <img src="images/6.png" alt="drawing" width="50"/> |     0     |           0          |     0    |    0   |
# | <img src="images/7.png" alt="drawing" width="50"/> |     1     |           0          |     1    |    1   |
# | <img src="images/8.png" alt="drawing" width="50"/> |     0     |           1          |     0    |    1   |
# | <img src="images/9.png" alt="drawing" width="50"/> |     1     |           0          |     0    |    0   |
# 
# 
# Therefore,
# - `X_train` contains three features for each example 
#     - Brown Color (A value of `1` indicates "Brown" cap color and `0` indicates "Red" cap color)
#     - Tapering Shape (A value of `1` indicates "Tapering Stalk Shape" and `0` indicates "Enlarging" stalk shape)
#     - Solitary  (A value of `1` indicates "Yes" and `0` indicates "No")
# 
# - `y_train` is whether the mushroom is edible 
#     - `y = 1` indicates edible
#     - `y = 0` indicates poisonous

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:",type(X_train))

print("First few elements of y_train:", y_train[:5])
print("Type of y_train:",type(y_train))

print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))


def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    entropy = 0.
    
    p = np.mean(y)
    
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1- p)*np.log2(1 - p)   
    
    return entropy

# Entropy at root should be 1
print("Entropy at root node: ", compute_entropy(y_train)) 

compute_entropy_test(compute_entropy)

# Split dataset
def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """
    
    left_indices = []
    right_indices = []
    
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
        
    return left_indices, right_indices

# Case 1

root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# The dataset only has three features, so this value can be 0 (Brown Cap), 1 (Tapering Stalk Shape) or 2 (Solitary)
feature = 0

left_indices, right_indices = split_dataset(X_train, root_indices, feature)

print("CASE 1:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# Visualize the split 
generate_split_viz(root_indices, left_indices, right_indices, feature)

print()

# Case 2

root_indices_subset = [0, 2, 4, 6, 8]
left_indices, right_indices = split_dataset(X_train, root_indices_subset, feature)

print("CASE 2:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# Visualize the split 
generate_split_viz(root_indices_subset, left_indices, right_indices, feature)

split_dataset_test(split_dataset)

# Calculate information gain
def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        cost (float):        Cost computed
    
    """    
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    information_gain = 0
    
    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0
    node_entropy = compute_entropy(y_node)
    
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    
    w_left = len(X_left)/len(X_node)
    w_right = len(X_right)/len(X_node)
    
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
   
    information_gain = node_entropy - weighted_entropy
    
    return information_gain

info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)

info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

# Get best split
def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    num_features = X.shape[1]
    
    best_feature = -1
    
    best_info_gain = 0
    
    for feature in range(num_features):
    # Compute information gain for this feature
        info_gain = compute_information_gain(X, y, node_indices, feature)
        
        # Update best feature if this feature's info gain is greater than the current best
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
   
    return best_feature

tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
generate_tree_viz(root_indices, y_train, tree)