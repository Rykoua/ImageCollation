from bs4 import BeautifulSoup
import requests
from os.path import join, exists
from pathlib import Path
from tqdm import tqdm


def download_themorgan_manuscript(url, dest_path):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find_all("img", {"class": "img-responsive"})
    images = dict()
    for i, result in enumerate(results):
        if i != 0:
            src = result.get("src")
            name = src.replace("../icaimages/6/", "")
            url = src.replace("..", "http://ica.themorgan.org")
            images[name] = url

    Path(dest_path).mkdir(parents=True, exist_ok=True)
    if len(images) > 0:
        for name, url in tqdm(images.items()):
            file_path = join(dest_path, name)
            if not(exists(file_path)):
                r = requests.get(url, allow_redirects=True)
                open(file_path, 'wb').write(r.content)


def download_file_url(url, dest_path):
    r = requests.get(url, allow_redirects=True)
    name = url.split("/")[-1]
    file_path = join(dest_path, name)
    open(file_path, 'wb').write(r.content)