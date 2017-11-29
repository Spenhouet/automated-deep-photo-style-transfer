import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--content_image_path", type=str, help="content image path")
    parser.add_argument("--style_image_path", type=str, help="style image path")

    args = parser.parse_args()

    if not os.path.isfile(args.content_image_path):
        print("File %s does not exist." % args.content_image_path)

    if not os.path.isfile(args.style_image_path):
        print("File %s does not exist." % args.style_image_path)
