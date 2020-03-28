import requests
import tarfile
import zipfile
import gzip
import shutil
import os
from nets.utils import progress_bar


def download_from_url(url, save_path, chunk_size=128):
    """Download a file from an URL.

    Original answer from https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url

    Args:
        url (str): path to the URL.
        save_path (str): path to the saving directory.
        chunk_size (int): download chunk.

    Returns:
        None
    """
    response = requests.get(url, stream=True)
    total = response.headers.get('content-length')
    with open(save_path, 'wb') as f:
        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                progress_bar(downloaded, total, "Downloading...")


def extract_to_dir(filename, dirpath='.'):
    # Does not create folder twice with the same name
    name, ext = os.path.splitext(filename)
    # if os.path.basename(name) == os.path.basename(dirpath):
    #     dirpath = '.'
    # Extract
    print(dirpath)
    print("Extracting...", end="")
    if tarfile.is_tarfile(filename):
        tarfile.open(filename, 'r').extractall(dirpath)
    elif zipfile.is_zipfile(filename):
        zipfile.ZipFile(filename, 'r').extractall(dirpath)
    elif ext == '.gz':
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        shutil.move(filename, os.path.join(dirpath, os.path.basename(filename)))
        print(f" | NOTE: gzip files are not extracted, and moved to {dirpath}", end="")
    # Return the path where the file was extracted
    print(" | Done !")
    return os.path.abspath(dirpath)
