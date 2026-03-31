"""
conftest.py — shared pytest fixtures and path setup for Project DiMaggio tests.
Adds the project root to sys.path so all packages are importable without install.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
