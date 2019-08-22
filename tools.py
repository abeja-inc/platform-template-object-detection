import os
import urllib


def download(url: str, filename: str) -> None:
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
