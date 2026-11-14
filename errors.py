"""Errors that stop a model-driven task without hiding the original failure."""


class ModelFailure(Exception):
    """Stop the job on a model failure; do not disguise it as a fetch failure."""
