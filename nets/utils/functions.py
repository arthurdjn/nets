"""
Utility functions.
"""


def append2dict(dict1, *dicts):
    """
    Append key values to another dict with the same keys.

    Args:
    dict1 (dict): dictionary where values will be added.
    dict2 (dict) dictionaries to extract values and append to another one.
        This dictionary should have the same keys as dict1.

    Returns:
        None

    Examples:

        >>> dict1 = {"key1": [], "key2": []}
        >>> dict2 = {"key1": 0, "key1": 1}
        >>> append2dict(dict1, dict2)
        >>> dict1
            {"key1": [0], "key2": [1]}

        >>> dict3 = {"key1": 2, "key1": 3}
        >>> dict4 = {"key1": 4, "key1": 5}
        >>> append2dict(dict1, dict3, dict4)
        >>> dict1
            {"key1": [0, 2, 4], "key2": [1, 3, 5]}
    """
    # Multiples dictionaries to merge
    for d in dicts:
        for (key, value) in d.items():
            # Test if the dictionary to append have the key
            try:
                dict1[key].append(value)
            # If not, create the key and merge the value
            except:
                dict1[key] = [value]