from abc import ABC

from fibad.plugin_utils import update_registry


# Make the type checker happy, all verbs have a base class
# xcxc Should this have base implementations? (See search.py for notes)
class Verb(ABC):
    pass


VERB_REGISTRY: dict[str, type[Verb]] = {
    "train": None,
    "infer": None,
    "download": None,
    "prepare": None,
    "rebuild_manifest": None,
}


def fibad_verb(cls: type[Verb]) -> type[Verb]:
    """_summary_"""
    update_registry(VERB_REGISTRY, cls.cli_name, cls)


def all_verbs() -> list[str]:
    return [verb for verb in VERB_REGISTRY if VERB_REGISTRY.get(verb) is not None]


def all_class_verbs() -> list[str]:
    return list(VERB_REGISTRY.keys())


def is_verb_class(cli_name: str) -> bool:
    return cli_name in VERB_REGISTRY


def fetch_verb_class(cli_name: str) -> type[Verb]:
    return VERB_REGISTRY.get(cli_name)
