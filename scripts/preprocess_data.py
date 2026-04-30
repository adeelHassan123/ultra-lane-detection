import argparse

from data.preprocess import copy_images, dilate_masks
from data.validate import validate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    copy_images(args.src, args.dst)
    dilate_masks(args.src, args.dst)
    validate_dataset(args.dst)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
