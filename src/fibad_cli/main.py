import argparse
import importlib
import sys
from importlib.metadata import version

# TODO config system, for now edit the dict

download_config = {
    "sw": "22asec",
    "sh": "22asec",
    "filter": ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
    "type": "coadd",
    "rerun": "pdr3_wide",
    "username": "mtauraso@local",
    "password": "cCw+nX53lmNLHMy+JbizpH/dl4t7sxljiNm6a7k1",
    "max_connections": 2,
    "fits_file": "../hscplay/temp.fits",
    "cutout_dir": "../hscplay/cutouts/",
    "offset": 0,
    "num_sources": 10,
}

config = {"download": download_config}


def main():
    """Primary entry point for the Fibad CLI. This handles dispatching to the various
    Fibad actions. The actions are defined in the pyproject.toml project.scripts
    section.
    """

    description = "Fibad CLI"
    epilog = "FIBAD is the Framework for Image-Based Anomaly Detection"

    #! We could potentially make this dynamic
    #! Somewhat difficult (perhaps impossible) to get this list from importlib/pkglib given
    #! That a fibad verb is simply an object in a file.
    fibad_verbs = ["train", "predict", "download"]

    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    parser.add_argument("--version", dest="version", action="store_true", help="Show version")
    parser.add_argument("-c", "--runtime-config", type=str, help="Full path to runtime config file")

    parser.add_argument("verb", nargs="?", choices=fibad_verbs, help="Verb to execute")

    args = parser.parse_args()

    print(f"Runtime config: {args.runtime_config}")

    if args.version:
        print(version("fibad"))
        return

    if not args.verb:
        parser.print_help()
        sys.exit(1)

    fibad_action = f"fibad-{args.verb}"

    # Ensure the action is available
    if args.verb not in fibad_verbs:
        print(f"Error: '{fibad_action}' is not available. Available actions are : {', '.join(fibad_verbs)}")
        sys.exit(1)

    importlib.import_module(f"fibad.{args.verb}").run(args, config)


if __name__ == "__main__":
    main()
