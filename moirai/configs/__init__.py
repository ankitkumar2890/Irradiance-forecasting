"""Self-contained configs for the three Moirai fine-tuning methods.

Each module here owns *only* the constants that ``moirai/moirai.py``,
``moirai/functions/preprocess.py``, ``moirai/functions/model.py`` and
``moirai/functions/results.py`` actually read. There are no API keys,
no station-grid math, no path-builder helpers - the original
``phase*/config.py`` files keep all of that for their own data-fetching
scripts. This split lets the ``moirai/`` pipeline run as a fully
self-contained tree (configs + code + data + outputs) without depending
on the sibling ``phase*/`` folders.
"""
