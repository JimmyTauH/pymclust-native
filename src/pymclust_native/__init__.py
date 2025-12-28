"""pymclust-native

Native Python reimplementation of mclust (Gaussian mixture modelling for clustering, classification, and density estimation).

Public API will mirror key mclust functions gradually.
"""

from .models import GMMResult, ModelName
from .em import GaussianMixtureEM
from .covariance import CovarianceModel

__all__ = [
    "GMMResult",
    "ModelName",
    "GaussianMixtureEM",
    "CovarianceModel",
]
