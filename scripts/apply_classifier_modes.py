#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MARKER='# CLASSIFIER_PHASE_MODES_V1'
def cell(path):
 src=f'''{MARKER}
TRAIN_MODE = "auto"       # auto | run | skip
VALIDATION_MODE = "auto"  # auto | run | skip | recompute
LOCKED_TEST_MODE = "manual"
from classifier_phase_planner import classifier_plan
CLASSIFIER_PLAN = classifier_plan(PROJECT_ROOT, PROJECT_ROOT / {path.as_posix()!r}, TRAIN_MODE, VALIDATION_MODE, LOCKED_TEST_MODE)
print(CLASSIFIER_PLAN)
'''
 return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'id':hashlib.sha256(path.as_posix().encode()).hexdigest()[:8],'source':src.splitlines(True)}
def main():
 for path in sorted((ROOT/'notebooks/3_classifiers').glob('0[1-4][a-j]_*.ipynb')):
  if 'LockedFinalTest' in path.name: continue
  nb=json.loads(path.read_text()); nb['cells']=[c for c in nb['cells'] if MARKER not in ''.join(c.get('source',[]))]
  insert=next((i+1 for i,c in enumerate(nb['cells']) if 'PROJECT_ROOT =' in ''.join(c.get('source',[]))),3)
  nb['cells'].insert(insert,cell(path.relative_to(ROOT))); path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n'); print(path.relative_to(ROOT))
if __name__=='__main__': main()
