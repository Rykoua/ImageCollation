import requests
from os.path import join, exists
import os
import fitz
from tqdm import tqdm
from pathlib import Path
import tempfile


def download_pdf(url, folder, name):
    r = requests.get(url, allow_redirects=True)
    file_path = join(folder, name + ".pdf")
    open(file_path, 'wb').write(r.content)
    return file_path


def download_pdf_to_temp(url):
    new_file, filename = tempfile.mkstemp()
    r = requests.get(url, allow_redirects=True)
    os.write(new_file, r.content)
    return new_file, filename


def save_pdf_image(file_path, dest_path):
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(file_path)
    i = 1
    images_name = list()
    xrefs = sorted([xref[0] for xref in doc.getPageImageList(0) if not(xref[0] in [10, 25, 26])])
    maximum_digits = len(str(len(xrefs)*3))
    for xref in tqdm(xrefs):
        pix = fitz.Pixmap(doc, xref)
        index = f'{i:0{maximum_digits}}'
        img_name = "image--{}.jpg".format(index)
        img_path = join(dest_path, img_name)
        if not(exists(img_path)):
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.writeImage(img_path)
            images_name.append(xref)
        i += 3


def pdf_2_images(url, dest_path):
    new_file, filename = download_pdf_to_temp(url)
    save_pdf_image(filename, dest_path)
    os.close(new_file)