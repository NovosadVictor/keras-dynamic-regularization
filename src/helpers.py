import os


def make_sure_dir_exists(path: str) -> str:
    if not os.path.isdir(path):
        os.makedirs(path)

    return path
