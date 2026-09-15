"""Generate the exact current shirt and normalize human metadata using bundled USD."""
import json
import os
from pathlib import Path
import subprocess
import sys
from pxr import Usd, UsdGeom

root = Path('/workspace/PhyRC_Sim')
environment = dict(os.environ)
environment.update(json.loads(Path('/project/config/geometry.json').read_text()))
source = '/project/assets/custom/garments/N_PR_2000_TShirt001.usd'
destination = root / 'Assets/Garment/Tops/Modelink'
destination.mkdir(parents=True, exist_ok=True)
for name, extra in [('t_shirt.usd', []), ('t_shirt_short.usd', ['0.7'])]:
    subprocess.run([sys.executable, '/scripts/make_wearable_shirt.py', source,
                    str(destination / name), *extra], env=environment, check=True)
for relative in ['Assets/Human/Mesh/manikin_exports/female2_c4-c5.usd',
                 'Assets/Material/Garment/linen_Pumpkin.usd']:
    stage = Usd.Stage.Open(str(root / relative))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().Save()
print('Prepared T-sleeve / collar-preserving 10/12-height runtime inputs.', flush=True)
