"""Registry-backed training/validation decisions; locked test remains manual."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

def _sha(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 return h.hexdigest()

def classifier_plan(project_root: str|Path, notebook: str|Path, train_mode='auto', validation_mode='auto', locked_test_mode='manual'):
 root=Path(project_root); rel=Path(notebook).resolve().relative_to(root).as_posix()
 registry=json.loads((root/'configs/final_classifier_registry.json').read_text())
 entry=next((x for x in registry['experiments'] if x.get('training_notebook')==rel),None)
 if entry is None: return {'training':{'action':'run','reason':'notebook absent from registry'},'validation':{'action':'run','reason':'registry evidence absent'},'locked_test':{'action':'skip','reason':'manual gate'}}
 evidence=entry.get('checkpoint_evidence') or {}; path=root/evidence.get('path',entry.get('checkpoint_path',''))
 signature=evidence.get('signature') or {}; verified=bool(evidence.get('verified') and path.is_file() and (not signature.get('sha256') or _sha(path)==signature['sha256']))
 if train_mode=='skip' and not verified: raise RuntimeError(f"Cannot skip unverified checkpoint: {path}")
 training='skip' if verified and train_mode=='auto' else ('skip' if train_mode=='skip' else 'run')
 val_evidence=entry.get('validation_threshold_evidence') or {}; validation_verified=bool(val_evidence.get('verified'))
 validation='skip' if validation_verified and validation_mode=='auto' else ('skip' if validation_mode=='skip' else 'run')
 if locked_test_mode!='manual': raise ValueError('LOCKED_TEST_MODE must remain manual in training notebooks')
 return {'training':{'action':training,'reason':'verified registry checkpoint' if verified else 'training_required'},
         'validation':{'action':validation,'reason':'verified validation threshold' if validation_verified else 'validation regeneration required'},
         'locked_test':{'action':'skip','reason':'manual locked-test gate'}}
