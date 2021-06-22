from os import listdir
from os.path import isfile, join
import hashlib


def get_dir_files(dir_path):
    return [f for f in listdir(dir_path) if isfile(join(dir_path, f))]


def get_md5_dir_dict(dir_path):
    files = get_dir_files(dir_path)
    return {f: hashlib.md5(open(join(dir_path, f),'rb').read()).hexdigest() for f in files}


def compare_dicts(dict1, dict2):
    keys1 = sorted(list(dict1.keys()))
    keys2 = sorted(list(dict2.keys()))
    if keys1 == keys2:
        for key in keys1:
            if dict1[key] != dict2[key]:
                return False
        return True
    else:
        return False



if __name__ == "__main__":
    md5_dict = get_md5_dir_dict("manuscripts/P1")
    md5_dict2 = get_md5_dir_dict("manuscripts/P1")
    md5_dict3 = get_md5_dir_dict("manuscripts/P2")
    exit()