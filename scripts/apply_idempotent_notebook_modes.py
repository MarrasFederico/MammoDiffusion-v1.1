#!/usr/bin/env python3
"""Deterministically add uniform phase modes and guard heavy notebook cells."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=sorted((ROOT/'notebooks/2_diffusers').glob('0[1-8]_*.ipynb'))
MARKER='# IDEMPOTENT_PHASE_MODES_V1'

def phase_for(source: str):
    executable=any(token in source for token in ('subprocess.Popen(', 'subprocess.run(', 'run_and_stream(', 'run_command('))
    if not executable: return None
    if any(x in source for x in ('train_ldm_v2.py','train_vae_v2.py','train_text_to_image')): return 'training'
    if any(x in source for x in ('generate_ldm_v2.py','sd_generation_worker.py')): return 'generation'
    if any(x in source for x in ('evaluate_filtered_ldm_v2.py','run_adaptive_filter.py')): return 'filter'
    if 'evaluate_ldm_v2.py' in source: return 'evaluation'
    return None

def mode_cell():
    source=f'''{MARKER}
TRAIN_MODE = "auto"       # auto | run | skip
GENERATION_MODE = "auto"  # auto | run | skip
EVALUATION_MODE = "auto"  # auto | run | skip | recompute
FILTER_MODE = "auto"      # auto | run | skip | recompute
PLAN_ONLY = False
ALLOW_HEAVY_RETRAIN = False       # must be True for auto mode to retrain from scratch
ALLOW_FULL_REGENERATION = False   # must be True for auto mode to regenerate a full image set

from artifact_phase_planner import plan_experiment, print_plan, phase_should_run
PHASE_MODES = {{"training": TRAIN_MODE, "generation": GENERATION_MODE,
               "evaluation": EVALUATION_MODE, "filter": FILTER_MODE}}
ALLOW_FLAGS = {{"training": ALLOW_HEAVY_RETRAIN, "generation": ALLOW_FULL_REGENERATION}}
PHASE_PLAN = plan_experiment(EXPERIMENT_DIR, PHASE_MODES, ALLOW_FLAGS)
print_plan(PHASE_PLAN)
'''
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'id':hashlib.sha256(MARKER.encode()).hexdigest()[:8],'source':source.splitlines(True)}

def main():
    for path in NOTEBOOKS:
        nb=json.loads(path.read_text()); cells=nb['cells']
        cells=[c for c in cells if MARKER not in ''.join(c.get('source',[]))]
        insert=next((i+1 for i,c in enumerate(cells) if 'EXPERIMENT_DIR =' in ''.join(c.get('source',[]))),3)
        cells.insert(insert,mode_cell())
        for cell in cells:
            if cell['cell_type']!='code': continue
            source=''.join(cell.get('source',[])); phase=phase_for(source)
            if not phase or source.startswith('# IDEMPOTENT_GUARD_V1'): continue
            indented=''.join(('    '+line if line.strip() else line) for line in source.splitlines(True))
            cell['source']=(f'# IDEMPOTENT_GUARD_V1:{phase}\nif phase_should_run(PHASE_PLAN, "{phase}", PLAN_ONLY):\n'+indented).splitlines(True)
            cell['execution_count']=None; cell['outputs']=[]
        nb['cells']=cells
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n")
        print(path.relative_to(ROOT))
if __name__=='__main__': main()
