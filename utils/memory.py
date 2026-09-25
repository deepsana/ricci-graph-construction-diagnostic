def rss_gb():
    """
    Resident set size of this process in GB, read from /proc (Linux only).

    :return: RSS in GB, or nan if /proc is not available
    """
    try:
        with open("/proc/self/status") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def frame_gb(df):
    """
    Memory held by the columns of a DataFrame.

    :param df: DataFrame to measure
    :return: size in GB, index excluded
    """
    return float(df.memory_usage(index=False).sum()) / 1e9

