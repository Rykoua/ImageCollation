from os.path import join, isfile, splitext, basename, dirname, exists
from os import listdir
from PIL import Image
import re


class DirTools:
    def __init__(self):
        pass

    @staticmethod
    def natural_sort(my_list):
        def convert(text):
            if text.isdigit():
                return int(text)
            else:
                return text.lower()

        def alphanum_key(key):
            return [convert(c) for c in re.split('([0-9]+)', key)]

        return sorted(my_list, key=alphanum_key)

    @staticmethod
    def list_folder_images(folder, natural_sort=True):
        image_types = ['jpg', 'tif', 'png', 'bmp']
        paths = [join(folder, f) for f in listdir(folder) if
                (isfile(join(folder, f)) and
                 (splitext(f)[1][1:].lower() in image_types))]
        if natural_sort:
            paths = DirTools.natural_sort(paths)
        return paths

    @staticmethod
    def get_annotation_path(file_path):
        file_name = basename(file_path)
        page_name_without_ext = splitext(file_name)[0].rsplit("_", 1)[0]
        annotation_name = page_name_without_ext + "_annotated" + splitext(basename(file_path))[1]
        annotation_folder_path = join(dirname(dirname(file_path)), "annotation")
        return join(annotation_folder_path, annotation_name)

    @staticmethod
    def get_annotation_name(annotation_path):
        page_name = splitext(basename(DirTools.get_annotation_path(annotation_path)))[0]
        page_name = page_name.split("_annotated")[0]
        return page_name

    @staticmethod
    def get_annotation_info(file_path):
        annotation_path = DirTools.get_annotation_path(file_path)
        annotation_name = DirTools.get_annotation_name(annotation_path)
        return annotation_name, annotation_path

    @staticmethod
    def get_images_list(folder):
        images_path = DirTools.list_folder_images(folder)
        return [Image.open(image_path).convert('RGB') for image_path in images_path]

    @staticmethod
    def get_annotation_dict(illustration_dir_path):
        illustration_paths = DirTools.list_folder_images(illustration_dir_path)
        annotation_dict = dict()
        for i, illustration_path in enumerate(illustration_paths):
            folio_name, annotation_path = DirTools.get_annotation_info(illustration_path)
            if not(annotation_path in annotation_dict):
                folio_file_name = folio_name + splitext(basename(annotation_path))[1]
                annotation_dict[annotation_path] = {"illustration": list(),
                                                    "folio": (folio_name, join(dirname(dirname(illustration_path)),
                                                                  folio_file_name))}

            annotation_dict[annotation_path]["illustration"].append((i, illustration_path))

        return annotation_dict

    @staticmethod
    def annotation_exists(illustration_dir_path):
        return exists(join(dirname(illustration_dir_path), "annotation"))

