"""A molecular property model built from first principles.

Nothing here wraps a cheminformatics library. SMILES parsing, circular
fingerprints, scaffold splitting and the network itself are all written out,
because the point is to understand what each step actually does -- and because
every one of them is a place where a subtle mistake produces a model that
scores well and predicts nothing.

Only numpy is required.
"""

__version__ = "0.1.0"
