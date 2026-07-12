from __future__ import annotations
import hashlib, json, tempfile, unittest
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'notebooks/utility'))
from artifact_phase_planner import load_runtime_manifest, plan_experiment, phase_should_run

class ArtifactPhasePlannerTests(unittest.TestCase):
 def fixture(self, root: Path, count=2):
  ck=root/'model.bin'; ck.write_bytes(b'checkpoint')
  images=root/'images'; images.mkdir()
  for i in range(count): (images/f'gen_{i:04d}.png').write_bytes(b'png'+bytes([i]))
  manifest={'schema_version':1,'provenance':'legacy_normalized_verified','phases':{
   'training':{'files':[{'path':'model.bin','size_bytes':ck.stat().st_size,'sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}]},
   'generation':{'image_sets':[{'path':'images','prefix':'gen_','start_index':0,'count':count}]},
   'evaluation':{'files':[{'path':'model.bin','sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}]},
   'filter':{'image_sets':[{'path':'images','prefix':'gen_','start_index':0,'count':count}]}}}
  (root/'legacy_runtime_manifest.json').write_text(json.dumps(manifest)); return manifest
 def test_complete_auto_skips_heavy_phases_and_is_idempotent(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); modes={'training':'auto','generation':'auto','evaluation':'auto','filter':'auto'}
   p1=plan_experiment(root,modes); p2=plan_experiment(root,modes)
   self.assertEqual(p1,p2); self.assertFalse(phase_should_run(p1,'training')); self.assertFalse(phase_should_run(p1,'generation'))
 def test_missing_metrics_only_recomputes_metrics(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root)
   manifest=json.loads((root/'legacy_runtime_manifest.json').read_text())
   manifest['phases']['evaluation']['files'][0]['path']='missing_metrics.json'
   (root/'legacy_runtime_manifest.json').write_text(json.dumps(manifest))
   p=plan_experiment(root,{'training':'auto','generation':'auto','evaluation':'auto'})
   self.assertEqual([x['action'] for x in p],['skip','skip','run'])
   p=plan_experiment(root,{'training':'auto','generation':'auto','evaluation':'recompute'})
   self.assertEqual([x['action'] for x in p],['skip','skip','recompute'])
 def test_partial_images_resume_generation_only(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); (root/'images/gen_0001.png').unlink()
   p=plan_experiment(root,{'training':'auto','generation':'auto'}); self.assertEqual([x['action'] for x in p],['skip','run'])
 def test_incompatible_checkpoint_never_skips(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); (root/'model.bin').write_bytes(b'changed')
   p=plan_experiment(root,{'training':'auto'}); self.assertEqual(p[0]['action'],'run')
 def test_present_directory_invalid_manifest_not_complete(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); (root/'legacy_runtime_manifest.json').write_text('{}')
   self.assertFalse(load_runtime_manifest(root)['valid'])
 def test_incompatible_legacy_provenance_isolated(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); x=json.loads((root/'legacy_runtime_manifest.json').read_text()); x['provenance']='legacy_unverified'; (root/'legacy_runtime_manifest.json').write_text(json.dumps(x))
   self.assertEqual(plan_experiment(root,{'training':'auto'})[0]['action'],'run')
 def test_historical_logs_are_never_touched(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); log=root/'energy.jsonl'; log.write_text('historic\n'); before=log.read_bytes(); plan_experiment(root,{'training':'auto','generation':'auto'}); self.assertEqual(before,log.read_bytes())
 def test_notebooks_expose_uniform_modes_and_guards(self):
  for p in sorted((ROOT/'notebooks/2_diffusers').glob('0[1-8]_*.ipynb')):
   text=p.read_text(); self.assertIn('IDEMPOTENT_PHASE_MODES_V1',text); self.assertIn('TRAIN_MODE',text); self.assertIn('GENERATION_MODE',text)
 def test_classifier_training_notebooks_keep_locked_test_manual(self):
  paths=sorted((ROOT/'notebooks/3_classifiers').glob('0[1-4][a-j]_*.ipynb'))
  for p in paths:
   if 'LockedFinalTest' in p.name: continue
   text=p.read_text(); self.assertIn('CLASSIFIER_PHASE_MODES_V1',text); self.assertIn('LOCKED_TEST_MODE',text); self.assertIn('manual',text)

if __name__=='__main__': unittest.main()
