from os.path import join, isfile, splitext
from os import listdir
from PIL import Image
import re

if __name__ == "__main__":
    def natural_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)


    def list_images_dir(dir_path):
        return natural_sort([f
                for f in listdir(dir_path)
                if (isfile(join(dir_path, f)) and
                    (splitext(f)[1][1:].lower() in ['jpg', 'png', 'bmp']))])

    def get_images_size(dir_path):
        return [Image.open(join(dir_path, f)).size for f in list_images_dir(dir_path)]


    pairs = [("P1", "P10"), ("P2", "P4_"), ("P3", "P7_"), ("D1", "D3__"), ("D2", "D4_"), ("D3", "D5_")]
    for pair in pairs:
        a = get_images_size("D:/Stage/tmp_manuscripts/{}".format(pair[1]))
        a0 = list_images_dir("D:/Stage/tmp_manuscripts/{}".format(pair[1]))
        b = get_images_size("D:/GitHub/ManuscriptTools/utils/manuscripts/{}".format(pair[0]))
        b0 = list_images_dir("D:/GitHub/ManuscriptTools/utils/manuscripts/{}".format(pair[0]))
        c = list()
        for i in range(len(a)):
            c.append((a[i][0]-b[i][0])**2 + (a[i][1]-b[i][1])**2)
        print("{} - {}:".format(pair[0], pair[1]))
        print(len([j for j in c if j != 0]))