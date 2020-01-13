import os


def makesure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return