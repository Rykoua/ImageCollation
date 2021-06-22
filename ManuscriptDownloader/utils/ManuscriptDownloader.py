import urllib.request
import json
from tqdm import tqdm
from PIL import Image, ImageOps
from os.path import join, exists
from pathlib import Path
from urllib import request
import math


class ManuscriptDownloader:
    def __init__(self):
        pass

    @staticmethod
    def get_json(url):
        with urllib.request.urlopen(url) as url_:
            data = json.loads(url_.read().decode())
            return data

    @staticmethod
    def crop_image(image_path, rgb=True, pixels_to_crop=54):
        im = Image.open(image_path)
        width, height = im.size
        area = (0, 0, width, height - pixels_to_crop)
        cropped_im = im.crop(area)
        if not (rgb):
            cropped_im = ImageOps.grayscale(cropped_im)
        cropped_im.save(image_path)

    @staticmethod
    def download_manuscript(ms_name, type, dest_path, width=763, height='', add_manuscript_name=False,
                            rgb=True, prefix="", pixels_to_crop=0, with_real_image_name=False,
                            divide_width_by=None):
        if type == "leipzig":
            img_urls = ManuscriptDownloader.get_leipzig_img_urls(ms_name, dest_path, prefix, width=width,
                                                                 height=height)
        elif type == "vatlib":
            img_urls = ManuscriptDownloader.get_vatlib_img_urls(ms_name, dest_path, prefix, width=width,
                                                                height=height,
                                                                with_real_image_name=with_real_image_name,
                                                                divide_width_by=divide_width_by)
        elif type == "wdl":
            img_urls = ManuscriptDownloader.get_wdl_img_urls(ms_name, dest_path)
        else:
            img_urls = ManuscriptDownloader.get_vatlib_img_urls(ms_name, dest_path, prefix, width=width,
                                                                height=height)

        if add_manuscript_name:
            dest_path = join(dest_path, ms_name)

        ManuscriptDownloader.download_images(img_urls, dest_path, rgb, pixels_to_crop)

    @staticmethod
    def get_vatlib_img_urls(ms_name, dest_path, prefix='', width=763, height='', with_real_image_name=False,
                            divide_width_by=None):
        manifest = "https://digi.vatlib.it/iiif/{}/manifest.json".format(ms_name)
        data = ManuscriptDownloader.get_json(manifest)
        img_urls = dict()
        # img_jsons = [canvas['@id'] for canvas in data['sequences'][0]['canvases']]
        urls = [canvas['images'][0]['resource']['service']['@id'] for canvas in
                data['sequences'][0]['canvases']]
        widths = [canvas['width'] for canvas in data['sequences'][0]['canvases']]
        resolutions = list()
        if width == -1:
            resolution = 'full'
        else:
            resolution = str(width) + ',' + str(height)
        if not (divide_width_by is None):
            resolutions = [str(math.ceil(width / divide_width_by)) + ',' for width in widths]
        maximum_digits = len(str(len(urls)))
        for i, url in enumerate(urls):
            img_name = url.split('/')[-1]
            if not (divide_width_by is None):
                resolution = resolutions[i]
            img_url = "https://digi.vatlib.it/pub/digit/{}/iiif/{}/full/{}/0/native.jpg".format(ms_name,
                                                                                                img_name,
                                                                                                resolution)
            if with_real_image_name:
                real_image_name = ".".join(img_name.split(".")[:-1]) + "_m"
                img_name = real_image_name + ".jpg"
            else:
                img_name = f'{i+1:0{maximum_digits}}.jpg'
                img_name = prefix + img_name
            img_path = join(dest_path, img_name)
            if not (exists(img_path)):
                img_urls[img_path] = img_url
        return img_urls

    @staticmethod
    def get_leipzig_img_urls(ms_name, dest_path, prefix='', width=763, height=''):
        manifest = "https://iiif.ub.uni-leipzig.de/{}/manifest.json".format(ms_name)
        data = ManuscriptDownloader.get_json(manifest)
        img_urls = dict()
        urls = [canvas['images'][0]['resource']['service']['@id'] for canvas in
                data['sequences'][0]['canvases']]
        if width == -1:
            resolution = 'full'
        else:
            resolution = str(width) + ',' + str(height)
        maximum_digits = len(str(len(urls)))
        for i, url in enumerate(urls):
            img_name = f'{i + 1:0{maximum_digits}}.jpg'
            img_name = prefix + img_name
            img_path = join(dest_path, img_name)
            img_url = url + "/full/{}/0/default.jpg".format(resolution)
            if not (exists(img_path)):
                img_urls[img_path] = img_url
        return img_urls

    @staticmethod
    def get_wdl_img_urls(ms_name, dest_path, width=1024, height=''):
        manifest = "https://www.wdl.org/en/item/{}/iiif/manifest.json".format(ms_name)
        data = ManuscriptDownloader.get_json(manifest)
        # canvases = [canvas["images"][0]["resource"]["service"]['@id'] for canvas in data['sequences'][0]['canvases']]
        urls = [canvas["images"][0]["resource"]['@id'] for canvas in data['sequences'][0]['canvases']]
        if width == -1:
            resolution = 'full'
        else:
            resolution = str(width) + ',' + str(height)
        img_urls = dict()
        for img_url in urls:
            image_id = img_url.split("/")[4]
            img_path = join(dest_path, image_id + ".jpg")
            img_url = "https://content.wdl.org/iiif/{}/full/{}/0/default.jpg".format(image_id, resolution)
            if not (exists(img_path)):
                img_urls[img_path] = img_url
        return img_urls

    @staticmethod
    def download_images(img_urls, ms_path, rgb, pixels_to_crop):
        Path(ms_path).mkdir(parents=True, exist_ok=True)
        if len(img_urls) > 0:
            for img_path, url in tqdm(img_urls.items()):
                urllib.request.urlretrieve(url, img_path)
                ManuscriptDownloader.crop_image(img_path, rgb, pixels_to_crop)


if __name__ == "__main__":
    manuscript_downloader = ManuscriptDownloader()
    ms_name = "10690"
    urls = manuscript_downloader.download_manuscript(ms_name, "wdl", "images3")
    exit()
