"""Python runs this at the start of every process of an engine launched with this dir on PYTHONPATH; with LAB_PROFILE
set, the process joins lab.profile's window."""
import os

if os.environ.get("LAB_PROFILE"):
    from lab.profile import window

    window.attach()
