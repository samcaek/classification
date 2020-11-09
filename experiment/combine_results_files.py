from os import listdir
from os.path import isfile, join


def main():
    """
    Combines the results files into one file so that you can easily copy it for further analysis.
    """

    results_path = '../data/results/'

    files = [f for f in listdir(results_path) if isfile(join(results_path, f))]

    lines = []

    all_results = 'all_results.csv'

    for name in files:
        if name == all_results:
            continue

        filename = results_path + name
        with open(filename, 'r') as file:
            lines += file.readlines()
            lines.append('\n\n')

    with open(results_path + all_results, 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':
    main()
