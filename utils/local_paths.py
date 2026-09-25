
def local_paths():
    """
    :return: the paths module of this machine (see paths_example.py)
    """
    try:
        import paths
    except ModuleNotFoundError:
        raise SystemExit("paths.py not found in the repo root: copy paths_example.py to paths.py and set the paths for this machine") from None
    return paths

