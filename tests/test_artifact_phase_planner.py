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
  manifest={'schema_version':1,'provenance':'runtime_assets_verified','phases':{
   'training':{'files':[{'path':'model.bin','size_bytes':ck.stat().st_size,'sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}]},
   'generation':{'image_sets':[{'path':'images','prefix':'gen_','start_index':0,'count':count}]},
   'evaluation':{'files':[{'path':'model.bin','sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}]},
   'filter':{'image_sets':[{'path':'images','prefix':'gen_','start_index':0,'count':count}]}}}
  (root/'runtime_manifest.json').write_text(json.dumps(manifest)); return manifest
 def test_complete_auto_skips_heavy_phases_and_is_idempotent(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); modes={'training':'auto','generation':'auto','evaluation':'auto','filter':'auto'}
   p1=plan_experiment(root,modes); p2=plan_experiment(root,modes)
   self.assertEqual(p1,p2); self.assertFalse(phase_should_run(p1,'training')); self.assertFalse(phase_should_run(p1,'generation'))
 def test_missing_metrics_only_recomputes_metrics(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root)
   manifest=json.loads((root/'runtime_manifest.json').read_text())
   manifest['phases']['evaluation']['files'][0]['path']='missing_metrics.json'
   (root/'runtime_manifest.json').write_text(json.dumps(manifest))
   p=plan_experiment(root,{'training':'auto','generation':'auto','evaluation':'auto'})
   self.assertEqual([x['action'] for x in p],['skip','skip','run'])
   p=plan_experiment(root,{'training':'auto','generation':'auto','evaluation':'recompute'})
   self.assertEqual([x['action'] for x in p],['skip','skip','recompute'])
 def test_frozen_evaluation_defers_to_notebook_selection_validation(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root)
   manifest=json.loads((root/'runtime_manifest.json').read_text())
   manifest['phases']['evaluation']['files'][0]['path']='legacy_metrics.json'
   (root/'runtime_manifest.json').write_text(json.dumps(manifest))
   plan=plan_experiment(root,{'evaluation':'frozen'})
   self.assertEqual(plan[0]['status'],'frozen_selection')
   self.assertEqual(plan[0]['action'],'skip')
   self.assertFalse(phase_should_run(plan,'evaluation'))
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
   root=Path(t); (root/'runtime_manifest.json').write_text('{}')
   self.assertFalse(load_runtime_manifest(root)['valid'])
 def test_incompatible_provenance_isolated(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); x=json.loads((root/'runtime_manifest.json').read_text()); x['provenance']='unverified'; (root/'runtime_manifest.json').write_text(json.dumps(x))
   self.assertEqual(plan_experiment(root,{'training':'auto'})[0]['action'],'run')
 def test_historical_logs_are_never_touched(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); log=root/'energy.jsonl'; log.write_text('historic\n'); before=log.read_bytes(); plan_experiment(root,{'training':'auto','generation':'auto'}); self.assertEqual(before,log.read_bytes())
 def test_notebooks_expose_uniform_modes_and_guards(self):
  for p in sorted((ROOT/'notebooks/2_diffusers').glob('0[1-8]_*.ipynb')):
   text=p.read_text(); self.assertIn('IDEMPOTENT_PHASE_MODES_V1',text); self.assertIn('TRAIN_MODE',text); self.assertIn('GENERATION_MODE',text)
 def test_classifier_training_notebooks_are_parameterized_and_keep_test_separate(self):
  paths=[ROOT/'notebooks/04_classifiers/01_MaxViT512.ipynb',
         ROOT/'notebooks/04_classifiers/02_MammoFM.ipynb']
  for p in paths:
   text=p.read_text(); self.assertIn('CONDITION =',text); self.assertIn('SEED =',text)
   self.assertNotIn('run_classifier_locked_test.py',text)

 # --- ALLOW_HEAVY_RETRAIN / ALLOW_FULL_REGENERATION -------------------------------------------
 def test_auto_incomplete_training_blocked_without_allow_heavy_retrain(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); (root/'model.bin').write_bytes(b'changed')
   plan=plan_experiment(root,{'training':'auto'},{'training':False})
   self.assertEqual(plan[0]['action'],'blocked')
   self.assertIn('ALLOW_HEAVY_RETRAIN',plan[0]['reason'])
   with self.assertRaises(RuntimeError):
    phase_should_run(plan,'training')
 def test_auto_incomplete_training_runs_with_allow_heavy_retrain(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); (root/'model.bin').write_bytes(b'changed')
   plan=plan_experiment(root,{'training':'auto'},{'training':True})
   self.assertEqual(plan[0]['action'],'run')
   self.assertTrue(phase_should_run(plan,'training'))
 def test_auto_incomplete_generation_blocked_without_allow_full_regeneration(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root)
   for p in (root/'images').glob('*.png'): p.unlink()
   plan=plan_experiment(root,{'generation':'auto'},{'generation':False})
   self.assertEqual(plan[0]['action'],'blocked')
   self.assertIn('ALLOW_FULL_REGENERATION',plan[0]['reason'])
 def test_partial_images_resume_ignores_allow_full_regeneration(self):
  # A partial (resumable) index gap is not a "full regeneration": must still run even when blocked.
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); (root/'images/gen_0001.png').unlink()
   plan=plan_experiment(root,{'generation':'auto'},{'generation':False})
   self.assertEqual(plan[0]['action'],'run')
 def test_zero_images_blocked_without_allow_full_regeneration(self):
  # No valid images at all is a genuine from-scratch regeneration: must be gated.
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root)
   for p in (root/'images').glob('*.png'): p.unlink()
   plan=plan_experiment(root,{'generation':'auto'},{'generation':False})
   self.assertEqual(plan[0]['action'],'blocked')
   plan=plan_experiment(root,{'generation':'auto'},{'generation':True})
   self.assertEqual(plan[0]['action'],'run')
 def test_complete_phase_never_blocked_regardless_of_allow_flags(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root)
   plan=plan_experiment(root,{'training':'auto','generation':'auto'},{'training':False,'generation':False})
   self.assertEqual([x['action'] for x in plan],['skip','skip'])
 def test_evaluation_and_filter_recompute_never_blocked(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root)
   manifest=json.loads((root/'runtime_manifest.json').read_text())
   manifest['phases']['evaluation']['files'][0]['path']='missing_metrics.json'
   (root/'runtime_manifest.json').write_text(json.dumps(manifest))
   plan=plan_experiment(root,{'evaluation':'recompute','filter':'recompute'},{})
   self.assertEqual([x['action'] for x in plan],['recompute','recompute'])
 def test_explicit_run_mode_bypasses_allow_heavy_retrain_gate(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root)
   plan=plan_experiment(root,{'training':'run'},{'training':False})
   self.assertEqual(plan[0]['action'],'run')
 def test_plan_experiment_without_allow_flags_is_unrestricted_backward_compatible(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); self.fixture(root); (root/'model.bin').write_bytes(b'changed')
   plan=plan_experiment(root,{'training':'auto'})
   self.assertEqual(plan[0]['action'],'run')
 def test_notebooks_expose_allow_heavy_flags(self):
  for p in sorted((ROOT/'notebooks/2_diffusers').glob('0[1-8]_*.ipynb')):
   text=p.read_text(); self.assertIn('ALLOW_HEAVY_RETRAIN',text); self.assertIn('ALLOW_FULL_REGENERATION',text)

if __name__=='__main__': unittest.main()
