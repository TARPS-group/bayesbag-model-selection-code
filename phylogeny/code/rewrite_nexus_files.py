import os
import sys
import argparse
from nexus import NexusReader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', metavar='dir')
    parser.add_argument('files', metavar='file', nargs='+')
    return parser.parse_args()

def rewrite_nexus_file(f, output_dir):
    fname = os.path.basename(f)
    nr = NexusReader(f)
    nr.write_to_file(os.path.join(output_dir, fname))


def main():
    args = parse_args()
    output_dir = args.output
    if not os.path.isdir(output_dir):
        sys.exit('output must be a directory')
    for f in args.files:
        print(f)
        rewrite_nexus_file(f, output_dir)

if __name__ == '__main__':
    main()
