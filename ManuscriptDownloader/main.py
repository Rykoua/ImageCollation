from utils.get_data import download_data, extract_illustrations
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find same illustrations in two different manuscripts')
    subparsers = parser.add_subparsers(dest='command')

    parser_download = subparsers.add_parser('download', help='download the manuscripts folios')
    parser_download.add_argument("-p", "--path", type=str,
                                 help='the path to the directory where the manuscripts will be saved')

    parser_extract = subparsers.add_parser('extract',
                                           help='extract illustrations from the manuscripts folios and store them')
    parser_extract.add_argument("-p", "--path", type=str,
                                help='the path to the directory in which manuscript are stored')
    parser_extract.add_argument("-a", "--annotation_path", type=str,
                                help='the path to the directory in which annotation files (JSON) are stored')
    args = parser.parse_args()

    if args.command == "download":
        print("=" * 10 + " Downloading folios " + "=" * 10)
        download_data(args.path)
    if args.command == "extract":
        print("=" * 10 + " Extracting illustrations " + "=" * 10)
        extract_illustrations(args.path, args.annotation_path)

