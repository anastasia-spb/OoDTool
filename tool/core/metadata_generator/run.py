import argparse
import os

from tool.core.metadata_generator.generator import generate_metadata


def run():
    parser = argparse.ArgumentParser(description='Generate metadata')
    parser.add_argument('-f', '--description_file', default='', required=False)
    args = parser.parse_args()

    output_folder, _ = os.path.split(args.description_file)

    meta_file_path = generate_metadata(args.description_file, output_folder)
    print(meta_file_path)


if __name__ == "__main__":
    run()
