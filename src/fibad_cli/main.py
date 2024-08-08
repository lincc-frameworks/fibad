import argparse
import shutil
import subprocess
import sys
from importlib.metadata import version


def main():
    """Primary entry point for the Fibad CLI. This handles dispatching to the various
    Fibad actions. The actions are defined in the pyproject.toml project.scripts
    section.
    """

    description = "Fibad CLI"
    epilog = "FIBAD is the Framework for Image-Based Anomaly Detection"

    #! We could potentially make this dynamic
    fibad_verbs = ["train", "predict"]

    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("--version", dest="version", action="store_true", help="Show version")
    parser.add_argument("verb", nargs="?", choices=fibad_verbs, help="Verb to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the verb")

    args = parser.parse_args()

    if args.version:
        print(version("fibad"))
        return

    if not args.verb:
        parser.print_help()
        sys.exit(1)

    fibad_action = f"fibad-{args.verb}"

    # Ensure the action is available
    if not shutil.which(fibad_action):
        print(f"Error: '{fibad_action}' is not available.")
        print("Is the action defined in the pyproject.toml project.scripts section?")
        sys.exit(1)

    # Execute the action with the remaining arguments
    try:
        result = subprocess.run([fibad_action] + args.args, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{fibad_action}' failed with exit code {e.returncode}.")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
