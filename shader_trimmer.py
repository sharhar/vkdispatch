import sys
import os

def trim_file(input_filename):
    output_filename = os.path.splitext(input_filename)[0] + '_trimmed.txt'
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(line[6:])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_filename>")
        sys.exit(1)
    trim_file(sys.argv[1])