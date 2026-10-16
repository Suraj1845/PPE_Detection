"""Backward-compatible entry point: ``python main.py [options]`` == ``python -m ppe_detection [options]``.

The original script ran the unverified, Ultralytics-based ``ppe.pt`` on a
hard-coded sample video. It has been replaced by the ``ppe_detection``
package; see README.md and MODEL_LICENSES.md.
"""

from ppe_detection.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
