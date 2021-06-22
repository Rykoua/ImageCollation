from PIL import Image, ImageDraw
import json
from os.path import join, splitext, isdir
from os import mkdir
from tqdm import tqdm
from pathlib import Path


class ExtractIllustrations:
    def __init__(self):
        pass

    @staticmethod
    def fix_area(area, im):
        x1, y1, x2, y2 = area
        if x1 < 0:
            x1 = 0
        if x2 >= im.width:
            x2 = im.width - 1
        if y1 < 0:
            y1 = 0
        if y2 >= im.height:
            y2 = im.height
        area = (x1, y1, x2, y2)
        return area

    @staticmethod
    def crop_image(image_path, dest_image_path, area):
        im = Image.open(image_path)
        area = ExtractIllustrations.fix_area(area, im)
        cropped_im = im.crop(area)
        cropped_im.save(dest_image_path)

    @staticmethod
    def draw_annotation(im, area):
        img1 = ImageDraw.Draw(im)
        img1.rectangle(area, outline="green", width=4)

    @staticmethod
    def get_json_data(file):
        with open(file) as json_file:
            data = json.load(json_file)
        return data

    @staticmethod
    def compute_scaled_area(start, length, scale):
        x1 = start
        x2 = x1 + length
        m = (x1 + x2) / 2
        d = length / 2
        return int(m - scale * d), int(m + scale * d)

    @staticmethod
    def get_areas_to_crop(annotations, x_scale=1, y_scale=1):
        areas_to_crop = dict()
        for _, file in annotations["_via_img_metadata"].items():
            name = file["filename"]
            regions = file["regions"]
            areas = []
            for region in regions:
                x = region['shape_attributes']['x']
                y = region['shape_attributes']['y']
                width = region['shape_attributes']['width']
                height = region['shape_attributes']['height']
                x1, x2 = ExtractIllustrations.compute_scaled_area(x, width, x_scale)
                y1, y2 = ExtractIllustrations.compute_scaled_area(y, height, y_scale)
                area = (x1, y1, x2, y2)
                areas.append(area)
            areas_to_crop[name] = areas
        return areas_to_crop

    @staticmethod
    def process_images(areas_to_crop, dir_path, dest_path):
        print("saving cropped images ...")
        Path(join(dest_path, "annotation")).mkdir(parents=True, exist_ok=True)
        Path(join(dest_path, "illustration")).mkdir(parents=True, exist_ok=True)
        for filename, areas in tqdm(areas_to_crop.items()):
            filepath = join(dir_path, filename)
            im = Image.open(filepath)
            new_path = join(dest_path, filename)
            im.save(new_path)
            name_without_extension, extension = splitext(filename)
            annotated_name = name_without_extension + "_annotated" + extension
            annotation_path = join(dest_path, "annotation", annotated_name)
            i = 0
            for area in areas:
                dest_file_name = name_without_extension + "_" + str(i) + extension
                dest_file_path = join(dest_path, "illustration", dest_file_name)
                ExtractIllustrations.draw_annotation(im, area)
                ExtractIllustrations.crop_image(filepath, dest_file_path, area)
                i += 1
            im.save(annotation_path)

    @staticmethod
    def save_annotations(json_file, dir_path, dest_path=None):
        if dest_path is None:
            dest_path = dir_path
        annotations = ExtractIllustrations.get_json_data(json_file)
        areas_to_crop = ExtractIllustrations.get_areas_to_crop(annotations)
        ExtractIllustrations.process_images(areas_to_crop, dir_path, dest_path)


if __name__ == "__main__":
    extract_illustrations = ExtractIllustrations()
    json_file = "D3.json"
    extract_illustrations.save_annotations(json_file, dir_path="")
    exit()