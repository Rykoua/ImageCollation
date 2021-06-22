import json


def convert_keys_to_string(my_dict):
    new_dict = dict()
    for key, value in my_dict.items():
        new_dict[key[0] + "," + key[1]] = value
    return new_dict


def convert_keys_to_tuple(my_dict):
    new_dict = dict()
    for key, value in my_dict.items():
        new_key = tuple(key.split(","))
        new_dict[new_key] = value
    return new_dict


def convert_matches_to_tuple(my_dict):
    new_dict = dict()
    for key, value in my_dict.items():
        new_dict[key] = [tuple(v) for v in value]
    return new_dict


def save_dict_as_json(my_dict, json_file):
    my_dict = convert_keys_to_string(my_dict)
    with open(json_file, 'w') as fp:
        json.dump(my_dict, fp)


def load_json_as_dict(json_file):
    with open(json_file) as fp:
        my_dict = json.load(fp)
        my_dict = convert_keys_to_tuple(my_dict)
        return convert_matches_to_tuple(my_dict)


def load_json_as_list(match_json_file):
    with open(match_json_file) as fp:
        matches = json.load(fp)
        return [(match[0], match[1]) for match in matches]



