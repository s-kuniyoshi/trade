"""
Pytest configuration and shared fixtures.
"""

import sys
import os
from pathlib import Path

# Add parent directory (python/) to path for imports
python_path = Path(__file__).parent.parent
if str(python_path) not in sys.path:
    sys.path.insert(0, str(python_path))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(python_path)
