import argparse
import sys
from importlib.metadata import version

from fibad import Fibad


def main():
    """Primary entry point for the Fibad CLI. This handles dispatching to the various
    Fibad actions.
    """

    description = "Fibad CLI"
    epilog = "FIBAD is the Framework for Image-Based Anomaly Detection"

    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    parser.add_argument("--version", dest="version", action="store_true", help="Show version")
    parser.add_argument("-c", "--runtime-config", type=str, help="Full path to runtime config file")

    parser.add_argument("verb", nargs="?", choices=Fibad.verbs, help="Verb to execute")

    args = parser.parse_args()

    if args.version:
        print(version("fibad"))
        return

    if not args.verb:
        parser.print_help()
        sys.exit(1)

    fibad_instance = Fibad(config_file=args.runtime_config)
    getattr(fibad_instance, args.verb)()


if __name__ == "__main__":
    main()
