"""NexGen iMATCH service layer.

HTTP API, persistence, authentication, tenancy, and audit for the recognition
engine in ``nexgen_engine``. Run it with::

    uvicorn imatch_api.main:app --host 0.0.0.0 --port 8443

The engine package holds no HTTP, database, or auth concerns, so it can be
tested and benchmarked without any of this.
"""

__version__ = "1.0.0"
