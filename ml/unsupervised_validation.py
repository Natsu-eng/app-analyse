# E:\gemini\app-analyse\ml\unsupervised_validation.py

# CHANGE: Nouveau module pour les techniques de validation non supervisée.

import numpy as np
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

def find_optimal_k_elbow(X, k_range=range(1, 11), random_state=42):
    """
    Trouve le nombre optimal de clusters (k) en utilisant la méthode du coude (Elbow Method).
    Calcule l'inertie (Within-Cluster Sum of Squares) pour une plage de k.
    """
    logger.info(f"Début de la recherche du k optimal (Elbow Method) pour k dans {k_range}.")
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Heuristique simple pour trouver le "coude"
    # On cherche le point qui a la plus grande distance à la ligne joignant le premier et le dernier point.
    points = np.array([k_range, inertias]).T
    first_point = points[0]
    last_point = points[-1]
    
    # Vecteur de la ligne
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    # Vecteur de chaque point à first_point
    vec_from_first = points - first_point
    
    # Projection sur la ligne
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (len(k_range), 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    
    # Distance perpendiculaire
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    
    # Le k optimal est celui qui a la plus grande distance
    optimal_k_index = np.argmax(dist_to_line)
    optimal_k = k_range[optimal_k_index]
    
    logger.info(f"Le k optimal trouvé par la méthode du coude est : {optimal_k}")
    return optimal_k, inertias
