# No external libraries are allowed to be imported in this file
import os

import sklearn
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random

# 1. COSINE SIMILARITY:
def similarity_matrix(matrix, k=5, axis=0):
    """
    This function should contain the code to compute the cosine similarity
    (according to the formula seen at the lecture) between users (axis=0) or 
    items (axis=1) and return a dictionary where each key represents a user
    (or item) and the value is a list of the top k most similar users or items,
    along with their similarity scores.
    
    Args:
        matrix (pd.DataFrame) : user-item rating matrix (df)
        k (int): number of top k similarity rankings to return for each \
                    entity (default=5)
        axis (int): 0: calculate similarity scores between users \
                        (rows of the matrix), 
                    1: claculate similarity scores between items \
                        (columns of the matrix)
    
    Returns:
        similarity_dict (dictionary): dictionary where the keys are users 
        (or items) and the values are lists of tuples containing the most 
        similar users (or items) along with their similarity scores.

    Note that is NOT allowed to authomatically compute cosine similarity using
    an apposite function from any package, the computation should follow the 
    formula that has been discussed during the lecture and that can be found in
    the slides.

    Note that is allowed to convert the DataFrame into a Numpy array for 
    faster computation.
    """
    similarity_dict= {}
    similarity_scores = {}
    # TO DO: loop through each couple of entities to calculate their cosine 
    # similarity and store these results
    # TO DO: Handle the absence of ratings (missing values in the matrix)
    if axis==0:
        for i in range(matrix.shape[0]):
            for j in range(i+1, matrix.shape[0]):
                # Get the ratings for every possible pair of users i and j
                user_i_ratings = matrix.iloc[i].values
                user_j_ratings = matrix.iloc[j].values
                
                # Find the indices of the items that both users have rated
                common_indices = np.where(~np.isnan(user_i_ratings) & ~np.isnan(user_j_ratings))[0]

                if len(common_indices) > 0:
                    # Calculate the cosine similarity formula
                    dot_product = np.dot(user_i_ratings[common_indices], user_j_ratings[common_indices])
                    norm_i = np.linalg.norm(user_i_ratings[common_indices])
                    norm_j = np.linalg.norm(user_j_ratings[common_indices])
                    similarity = dot_product / (norm_i * norm_j)

                    # Store the similarity in the dictionary under the user IDs (not row positions)
                    user_i_id, user_j_id = matrix.index[i], matrix.index[j]
                    if user_i_id not in similarity_scores:
                        similarity_scores[user_i_id] = []
                    if user_j_id not in similarity_scores:
                        similarity_scores[user_j_id] = []

                    similarity_scores[user_i_id].append((user_j_id, similarity))
                    similarity_scores[user_j_id].append((user_i_id, similarity))

    # TO DO: If axis is 1, what do you need to do to calculate the similarity 
    # between items (columns)
    if axis==1:
        for i in range(matrix.shape[1]):
            for j in range(i+1, matrix.shape[1]):
                # Get the ratings for every possible pair of items i and j
                item_i_ratings = matrix.iloc[:, i].values
                item_j_ratings = matrix.iloc[:, j].values
                
                # Find the indices of the users that have rated both items
                common_indices = np.where(~np.isnan(item_i_ratings) & ~np.isnan(item_j_ratings))[0]

                if len(common_indices) > 0:
                    # Calculate the cosine similarity formula
                    dot_product = np.dot(item_i_ratings[common_indices], item_j_ratings[common_indices])
                    norm_i = np.linalg.norm(item_i_ratings[common_indices])
                    norm_j = np.linalg.norm(item_j_ratings[common_indices])
                    similarity = dot_product / (norm_i * norm_j)

                    # Store the similarity in the dictionary under the movie IDs (not column positions)
                    item_i_id, item_j_id = matrix.columns[i], matrix.columns[j]
                    if item_i_id not in similarity_scores:
                        similarity_scores[item_i_id] = []
                    if item_j_id not in similarity_scores:
                        similarity_scores[item_j_id] = []

                    similarity_scores[item_i_id].append((item_j_id, similarity))
                    similarity_scores[item_j_id].append((item_i_id, similarity))
    



    # TO DO: sort the similarity scores for each entity and add the top k most 
    # similar entities to the similarity_dict
    for entity, scores in similarity_scores.items():
        # Sort the scores in descending order and select the top k
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        similarity_dict[entity] = sorted_scores

    return similarity_dict

def loss_function(utility_matrix, user_matrix, item_matrix, regularization=0.02):
    """
    This function should contain the code to compute the loss function 
    (according to the formula seen at the lecture) for the matrix factorization
    algorithm.

    Args:
        utility_matrix (np.ndarray): user-item rating matrix
        user_matrix (np.ndarray): user matrix
        item_matrix (np.ndarray): item matrix

    Returns:
        float: the computed loss
    """
    # Compute the squared error for each observed rating
    squared_errors = np.where(~np.isnan(utility_matrix), (utility_matrix - np.dot(user_matrix, item_matrix.T)) ** 2, 0)
    
    # Sum the squared errors and add the regularization term
    loss = np.sum(squared_errors) + regularization * (np.sum(user_matrix ** 2) + np.sum(item_matrix ** 2))
    return loss

def gradient_loss_function(utility_matrix, user_matrix, item_matrix, regularization=0.02):
    """
    This function should contain the code to compute the gradient of the loss 
    function (according to the formula seen at the lecture) for the matrix 
    factorization algorithm.

    Args:
        utility_matrix (np.ndarray): user-item rating matrix
        user_matrix (np.ndarray): user matrix
        item_matrix (np.ndarray): item matrix
        regularization (float): regularization parameter

    Returns:
        tuple: the gradients of the loss function with respect to the user and item matrices
    """
    
    # Error on observed ratings only, 0 where the rating is missing
    error = np.where(~np.isnan(utility_matrix), utility_matrix - np.dot(user_matrix, item_matrix.T), 0)

    # Compute the gradient of the loss function with respect to the user matrix
    grad_user = -2 * np.dot(error, item_matrix) + 2 * regularization * user_matrix

    # Compute the gradient of the loss function with respect to the item matrix
    grad_item = -2 * np.dot(error.T, user_matrix) + 2 * regularization * item_matrix
    
    return grad_user, grad_item



# 2. COLLABORATIVE FILTERING
def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement user-based collaborative
    filtering, returning the predicted rate associated to a target user-movie
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        user_similarity (dict): dictonary containing user similarities, \
            obtained using the similarity_matrix function (axis=0)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to user-based \
        collaborative filtering
    """
    # TO DO: retrieve the topk most similar users for the target user
    similar_users = user_similarity.get(user_id, [])[:k]    #

    # TO DO: implement user-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    numerator = 0  
    denominator = 0  
    for neighbour_id, similarity in similar_users:
        # Rating of the neighbour for the target movie (NaN if not rated)
        rating = user_item_matrix.loc[neighbour_id, movie_id]
        # Only neighbours that rated the movie contribute to the prediction
        if not np.isnan(rating):
            numerator += similarity * rating
            denominator += similarity
    
    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement item-based collaborative
    filtering, returning the predicted rate associated to a target user-movie 
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        item_similarity (dict): dictonary containing item similarities, \
            obtained using the similarity_matrix function (axis=1)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to item-based 
        collaborative filtering
    """
    # TO DO: retrieve the topk most similar users for the target item
    similar_items = item_similarity.get(movie_id, [])[:k]

    # TO DO: implement item-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    numerator = 0  
    denominator = 0  
    for neighbour_id, similarity in similar_items:
        # Rating of the target user for the neighbouring movie (NaN if not rated)
        rating = user_item_matrix.loc[user_id, neighbour_id]
        # Only movies the user rated contribute to the prediction
        if not np.isnan(rating):
            numerator += similarity * rating
            denominator += similarity

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating

# 3. MATRIX FACTORIZATION
def matrix_factorization(
        utility_matrix: np.ndarray,
        feature_dimension=2,
        learning_rate=0.001,
        regularization=0.02,
        n_steps=2000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function should contain the code to implement matrix factorisation
    using the Gradient Descent with Regularization method (according to the psuedo code
    seen at the lecture), returning the user and item matrices.

    Args:
        utility_matrix (np.ndarray): user-item rating matrix
        feature_dimension (int): number of latent features (default=2)
        learning_rate (float): learning rate for gradient descent \
            (default=0.001)
        regularization (float): regularization parameter (default=0.02)
        n_steps (int): number of iterations for gradient descent \
            (default=2000)

    Returns:
        user_matrix (np.ndarray): user matrix
        item_matrix (np.ndarray): item matrix
    """

    user_matrix = np.random.rand(utility_matrix.shape[0], feature_dimension)
    item_matrix = np.random.rand(utility_matrix.shape[1], feature_dimension)

    for step in range(n_steps):
        # TODO: Implement the algorithm to update user_matrix and item_matrix
        grad_user,grad_item = gradient_loss_function(utility_matrix, user_matrix, item_matrix, regularization)
        user_matrix -= learning_rate * grad_user
        item_matrix -= learning_rate * grad_item

    return user_matrix, item_matrix


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data') 
    path =  os.path.join(DATA_DIR, "u.data")       
    df = pd.read_table(path, sep="\t", names=[
        "UserID", "MovieID", "Rating", "Timestamp"
    ])
    df = df.pivot_table(
        index = 'UserID', 
        columns = 'MovieID', 
        values = 'Rating'
    )

    # You can use this section for testing the similarity_matrix function: 
    # Return the top 5 most similar users to user 3:
    user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    print(user_similarity_matrix.get(3,[]))

    # Return the top 5 most similar items to item 10:
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    print(item_similarity_matrix.get(10,[]))

    
    # You can use this section for testing the user_based_cf and the 
    # item_based_cf functions: Return the predicted ratings assigned by user 
    # 13 to movie 100:
    user_id = 13  
    movie_id = 100  

    u_predicted_rating = user_based_cf(
        user_id, 
        movie_id, 
        user_similarity_matrix, 
        user_item_matrix = df,
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to user-based collaborative filtering is: "
        f"{u_predicted_rating:.2f}"
    )

    i_predicted_rating = item_based_cf(
        user_id,
        movie_id, 
        item_similarity_matrix,
        user_item_matrix = df, 
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to item-based collaborative filtering is: "
        f"{i_predicted_rating:.2f}"
    )

    utility_matrix = np.array([
        [5, 2, 4, 4, 3],
        [3, 1, 2, 4, 1],
        [2, np.nan, 3, 1, 4],
        [2, 5, 4, 3, 5],
        [4, 4, 5, 4, np.nan],
    ])
    user_matrix, item_matrix = matrix_factorization(
        utility_matrix, learning_rate=0.001, n_steps=5000
    )

    print("Current guess:\n", np.dot(user_matrix, item_matrix.T))
