"""
This modules defines basic function to render a simulation, like progress bar and statistics table.
"""

import time


def get_time(start_time, end_time):
    """Get ellapsed time in minutes and seconds.

    Args:
        start_time (float): strarting time
        end_time (float): ending time

    Returns:
        elapsed_mins (float): elapsed time in minutes
        elapsed_secs (float): elapsed time in seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def progress_bar(current_index, max_index, prefix=None, suffix=None, start_time=None):
    """Display a progress bar and duration.

    Args:
        current_index (int): current state index (or epoch number).
        max_index (int): maximal numbers of state.
        prefix (str, optional): prefix of the progress bar. The default is None.
        suffix (str, optional): suffix of the progress bar. The default is None.
        start_time (float, optional): starting time of the progress bar. If not None, it will display the time
            spent from the beginning to the current state. The default is None.

    Returns:
        None. Display the progress bar in the console.
    """
    # Add a prefix to the progress bar
    prefix = "" if prefix is None else str(prefix) + " "

    # Get the percentage
    percentage = current_index * 100 // max_index
    loading = "[" + "=" * (percentage // 2) + " " * (50 - percentage // 2) + "]"
    progress_display = "\r{0}{1:3d}% | {2}".format(prefix, percentage, loading)

    # Add a suffix to the progress bar
    progress_display += "" if suffix is None else " | " + str(suffix)

    # Add a timer
    if start_time is not None:
        time_min, time_sec = get_time(start_time, time.time())
        time_display = " | Time: {0}m {1}s".format(time_min, time_sec)
        progress_display += time_display

    # Print the progress bar
    # TODO: return a string instead
    print(progress_display, end="{}".format("" if current_index < max_index else " | Done !\n"))


def describe_stats(state_dict):
    """Describe and render a dictionary. Usually, this function is called on a ``Solver`` state dictionary,
    and merged with a progress bar.

    Args:
        state_dict (dict): the dictionary to showcase

    Returns:
        string: the dictionary to render.
    """
    stats_display = ""
    for idx, (key, value) in enumerate(state_dict.items()):
        if type(value) == float:
            if idx > 0:
                stats_display += " | "
            stats_display += f"{str(key).capitalize()[:4]}.: {value:.4f}"

    return stats_display
