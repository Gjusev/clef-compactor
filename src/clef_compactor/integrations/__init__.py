"""Optional integrations with orchestration frameworks.

Each module imports its framework lazily at import time, so ``clef_compactor``
itself stays dependency-light. Install the matching extra to use them::

    pip install clef-compactor[langchain]
    pip install clef-compactor[llamaindex]
"""
