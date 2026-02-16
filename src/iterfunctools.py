"""Some useful tools to work with iterables, collections & functions."""

from collections.abc import MutableMapping

from funcy import set_in


def unnest_dict(nested_dict: MutableMapping, parent_key="", separator="."):
    """Unnest a dictionary, flattenning it by joining keys with a separator.

    Args:
    ----
        nested_dict (dict): The dictionary to unnest.
        parent_key (str, optional): The parent key to use for nested dictionaries. Defaults to an empty string.
        separator (str, optional): The separator to use for joining keys. Defaults to ".".

    Returns:
    -------
        dict: The unnested dictionary.

    """
    items = []
    for key, value in nested_dict.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(unnest_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def nest_dict(flat_dict: MutableMapping, separator="."):
    """Nest a dictionary, unflattening it by splitting keys with a separator.

    Args:
    ----
        flat_dict (dict): The dictionary to nest.
        separator (str, optional): The separator to use for splitting keys. Defaults to ".".

    Returns:
    -------
        dict: The nested dictionary.

    """
    nested_dict = {}
    for k, v in flat_dict.items():
        nested_dict = set_in(nested_dict, k.split(separator), v)

    return nested_dict


def sort_keys(dic: dict):
    """Sort a dictionary by its keys.

    Args:
    ----
        dic (dict): The dictionary to sort.

    Returns:
    -------
        dict: The sorted dictionary.

    """
    return dict(sorted(dic.items()))
