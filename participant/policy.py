"""PhyRC official policy entrypoint (returns two robots x nine actions)."""
import sys
from pathlib import Path
PARTICIPANT=Path(__file__).resolve().parent
if str(PARTICIPANT) not in sys.path:sys.path.insert(0,str(PARTICIPANT))
from policyv2 import load_policy
