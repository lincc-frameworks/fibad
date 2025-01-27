import argparse
import sys
from importlib.metadata import version

from fibad import Fibad
from fibad.verbs import all_verbs, fetch_verb_class, is_verb_class


def main():
    """Primary entry point for the Fibad CLI. This handles dispatching to the various
    Fibad actions and returning a result.
    """

    description = "Fibad CLI"
    epilog = "FIBAD is the Framework for Image-Based Anomaly Detection"

    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    parser.add_argument("--version", dest="version", action="store_true", help="Show version")
    parser.add_argument("-c", "--runtime-config", type=str, help="Full path to runtime config file")

    # cut off "usage: " from beginning and "\n" from end so we get an invocation
    # which subcommand parsers can add to appropriately.
    subparser_usage_prefix = parser.format_usage()[7:-1]
    subparsers = parser.add_subparsers(title="Verbs:", required=False)

    # Add a subparser for every verb, (whether defined by function or class)
    for cli_name in all_verbs():
        print(cli_name)
        subparser_kwargs = {}

        if is_verb_class(cli_name):
            verb_class = fetch_verb_class(cli_name)
            subparser_kwargs = verb_class.add_parser_kwargs

        verb_parser = subparsers.add_parser(
            cli_name, prog=subparser_usage_prefix + " " + cli_name, **subparser_kwargs
        )

        if is_verb_class(cli_name):
            verb_class.setup_parser(verb_parser)

        verb_parser.set_defaults(verb=cli_name)

    args = parser.parse_args()

    if args.version:
        print(version("fibad"))
        return

    if not args.verb:
        parser.print_help()
        sys.exit(1)

    fibad_instance = Fibad(config_file=args.runtime_config)
    retval = 0
    if is_verb_class(args.verb):
        verb = fetch_verb_class(cli_name)(fibad_instance.config)
        retval = verb.run_cli(args)
    else:
        getattr(fibad_instance, args.verb)()

    exit(retval)


if __name__ == "__main__":
    main()
