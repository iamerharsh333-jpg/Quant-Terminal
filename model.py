"""Backward-compatible model API.

The production implementation now lives in ``models.training``. This wrapper
keeps existing imports such as ``from model import train_model`` working.
"""

from models.training import *  # noqa: F401,F403

