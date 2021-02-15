# File: _utils.py
# Creation: Wednesday August 19th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


def inv_permutation(permutation):
    """Get the inverse of a permutation. Used to invert a transposition for example.

    Args:
        permutation (list or tuple): permutation to invert.

    Returns:
        list
    """
    inverse = [0] * len(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i
    return inverse



