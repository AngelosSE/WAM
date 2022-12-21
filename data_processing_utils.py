import functools
import hashlib
import pickle
import numpy as np
import omegaconf
from definitions import Attribute, Data
from numpy.typing import NDArray

def project(
        data : Data, 
        target_attrs : list[Attribute]) -> Data:
    """
    This is not a proper projection, since in a projection after columns are 
    removed any non-unique row should be made unique. It is a projection if we 
    consider also the row index as an attribute.
    """

    if not set(target_attrs).issubset(set(data.keys())): # validate arguments
        raise ValueError(set(target_attrs)-set(data.keys()))

    return {attr:data[attr] for attr in target_attrs}

def restrict(
        data : Data, 
        restrictions : dict[Attribute,NDArray]) -> Data:

    restr_attrs = [*restrictions.keys()]
    if not set([*restrictions.keys()]).issubset(set(data.keys())): # validate arguments
        raise ValueError(set(restr_attrs)-set(data.keys()))

    n_data = list(data.values())[0].shape[0]
    admissible_samples = np.array(n_data*[True]) # assume every sample is admissible
    for attr,vals in data.items():
        admissible_vals = restrictions.get(attr)
        if admissible_vals is not None:
            admissible_samples = admissible_samples \
                                    & np.isin(vals,admissible_vals)

    return {k:v[admissible_samples] for k,v in data.items()} 

def select(
    data : Data,
    restrictions : dict[Attribute,NDArray],
    attributes : list[Attribute]) -> Data:

    return project(restrict(data,restrictions),attributes)

def restrict_indices(data,indices):
    """Restricts data to only samples with index in indices."""
    return {attr:vals[indices] for attr,vals in data.items()}