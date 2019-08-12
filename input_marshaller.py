import os
import os.path
import sys

def marshall(dir, out):
    """Reads the files in the given directory and writes a TSV containing (clue\tlen\tsoln)."""
    with open(out, 'w') as outfile:
        for filename in os.listdir(dir):
            infile = os.path.join(dir, filename)
            with open(infile, 'r') as data:
                for line in data:
                    clue = line.split(' : ')
                    if len(clue) == 2:
                        cl = clue[0].strip()
                        an = clue[1].strip()
                        outfile.write('{clue}\t{len}\t{ans}\n'
                            .format(clue=cl, len=len(an), ans=an))

if __name__ == '__main__':
    marshall(sys.argv[1], sys.argv[2])
