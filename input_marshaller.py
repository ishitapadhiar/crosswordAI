
def marshall(infile):
    """Reads the clue data from the given file and outputs a list of (clue, solution)."""
    ls = []
    with open(infile, 'r') as data:
        for line in data:
            clue = line.split(' : ')
            if len(clue) == 2:
                ls.append((clue[0].strip(), clue[1].strip()))
    return ls
