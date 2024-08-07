import argparse


def main():
    """Argument parser to process command line arguments when training with Fibad."""

    parser = argparse.ArgumentParser(description="Training with Fibad.")

    parser.add_argument("-c", "--runtime-config", type=str, help="Full path to runtime config file")

    args = parser.parse_args()

    run(args)


def run(args):
    """Note: Don't import anything from Fibad outside of this run function.
    Keeping all the imports inside the run function ensures that the --help option
    returns quickly without loading all the dependencies.
    """
    from fibad.train import run

    run(args)


if __name__ == "__main__":
    main()
