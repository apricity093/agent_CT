"""Lazy discovery of the public imaging domains, independent of example cases."""


def list_capabilities(modality=None):
    """Return all domains, or the same object as the selected domain's API."""
    from importlib import import_module

    domains = ('ct', 'mri', 'deconvolution')
    if modality is None:
        return {domain: list_capabilities(domain) for domain in domains}
    if modality not in domains:
        raise ValueError(f'Unknown modality: {modality!r}; expected one of {domains}')
    return import_module(f'inv_framework.{modality}.api').list_capabilities()
