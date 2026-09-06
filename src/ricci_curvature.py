import numpy as np
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import networkx as nx

def forman_ricci_curvature(G):
    '''
    Calculates curvature based on combinatorial discretization of Ricci curvature as follows
    \kappa(i,j) = 4 - d_i - d_j + 3 \triangle(i,j)
    where d_i and d_j are degree counts of node i and node j respectively, and
    \tiangle (i,j) is the number of triangles that ocntains edge (i,j)
    :param G: graph object
    :return: Graph with original info along with calculated FRC values
    '''
    frc = FormanRicci(G)
    frc.compute_ricci_curvature()
    G_frc = frc.G.copy()

    return G_frc

def ollivier_ricci_curvature(G):
    '''
    Calculates curvature based on optimal transport so by comparing distance betwwen 2 nodes to the Wassertein
    distance between their local neighborhood
    \kappa(i,j) = 1 - \frac{W_1(m_i,m_j)}{d(i,j)}
    where d(i,j) is the shortest distance between i and j, m_i is the probability distribution over the neighbors of node i
    :param G: graph object
    :return: Graph with original info along with calculated ORC values
    '''
    orc = OllivierRicci(G, alpha=0.5, method="Sinkhorn",proc=1, verbose="ERROR")
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()
    return G_orc