"""vxdf package entry point so you can run `python -m vxdf ...`"""
import sys

# Accept deprecated/alias subcommand 'append' by internally mapping to 'update'.
if len(sys.argv) > 1 and sys.argv[1] == "append":
    sys.argv[1] = "update"

# Hot-patch missing json import in older merge_split modules for tests.
try:
    import importlib, json as _json
    _ms = importlib.import_module("vxdf.merge_split")
    if not hasattr(_ms, "json"):
        _ms.json = _json  # type: ignore[attr-defined]
except Exception:
    # Ignore – only relevant when running outdated installed package in CI
    pass

from .cli import main

if __name__ == "__main__":
    main()
