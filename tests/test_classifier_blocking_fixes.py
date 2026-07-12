from __future__ import annotations
import json, os, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"notebooks/utility")); sys.path.insert(0,str(ROOT/"scripts"))
import classifier_interpretability as ci
import check_classifier_runtime_environment as envcheck
import locked_matrix_inference as locked
import classifier_dataset_builder as cdb


class InterpretabilityTests(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
    def test_spatial_and_mammofm_gradcam_are_finite(self):
        import torch
        class Encoder(torch.nn.Module):
            def __init__(self): super().__init__(); self._conv_head=torch.nn.Conv2d(1,2,1)
            def extract_features(self,x): return self._conv_head(x)
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.image_encoder=Encoder(); self.fc=torch.nn.Linear(2,1)
            def forward(self,x): return self.fc(self.image_encoder.extract_features(x).mean((2,3)))
        model=Model(); image=torch.randn(1,1,8,8,requires_grad=True)
        for heat in (ci.torch_spatial_gradcam(model,image,model.image_encoder._conv_head), ci.mammofm_gradcam(model,image)):
            self.assertEqual(heat.shape,(8,8)); self.assertTrue((heat>=0).all()); self.assertLessEqual(float(heat.max()),1)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
    def test_raddino_uses_patch_tokens_not_maxvit_api(self):
        import torch
        class TokenLayer(torch.nn.Module):
            def forward(self,x): return x*1.1
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.token=TokenLayer(); self.head=torch.nn.Linear(3,1)
            def forward(self,x): return self.head(self.token(x)[:,0])
        model=Model(); tokens=torch.randn(1,17,3,requires_grad=True)
        heat=ci.raddino_token_attribution(model,tokens,model.token)
        self.assertEqual(heat.shape,(4,4)); self.assertFalse(hasattr(model,"forward_features"))


class EnvironmentTests(unittest.TestCase):
    def test_notebook_names_are_not_model_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"results").mkdir(); (root/"configs").mkdir(); (root/"notebooks").mkdir()
            (root/"notebooks/RAD-DINO_Mammo-FM.ipynb").write_text("{}")
            with patch.dict(os.environ,{"HF_HOME":str(root/"empty-cache")},clear=False): report=envcheck.audit(root)
            self.assertIsNone(report["assets"]["raddino"]); self.assertIsNone(report["assets"]["mammofm"])


class DatasetLeakageTests(unittest.TestCase):
    def test_real_patient_overlap_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); metadata=root/"data/processed/metadata"; metadata.mkdir(parents=True)
            train_image=root/"data/processed/train/0/i1.png"; train_image.parent.mkdir(parents=True); train_image.write_bytes(b"x")
            val_image=root/"data/processed/val/0/i2.png"; val_image.parent.mkdir(parents=True); val_image.write_bytes(b"x")
            (metadata/"train.csv").write_text("patient_id,image_id,label,processed_path\np1,i1,0,data/processed/train/0/i1.png\n")
            (metadata/"val.csv").write_text("patient_id,image_id,label,processed_path\np1,i2,0,data/processed/val/0/i2.png\n")
            variant={"dataset_variant_id":"R","status":"ready","real_source":"data/processed/metadata/train.csv","augmentation_source":None,"synthetic_generator_id":None,"synthetic_count_by_class":{}}
            with self.assertRaisesRegex(RuntimeError,"patient leakage"): cdb.build_training_and_validation_rows(root,variant)


class LockedFixtureTests(unittest.TestCase):
    def test_complete_three_seed_fixture_and_one_shot_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); lock_dir=root/"results/final_evaluation_v2"; lock_dir.mkdir(parents=True)
            for panel in ("primary","secondary","ablation"):
                (lock_dir/f"{panel}_panel_manifest.json").write_text(json.dumps({"experiment_ids":["maxvit512__R"] if panel=="primary" else []}))
            (root/"data/processed/metadata").mkdir(parents=True); (root/"data/processed/metadata/test.csv").write_text("patient_id,image_id,label,processed_path\np1,i1,0,x\np2,i2,1,y\n")
            (root/"configs").mkdir(); jobs=[]
            base=root/"results/classifiers_matrix/maxvit512/R/maxvit512_standard/ensemble/metrics"; base.mkdir(parents=True)
            (base/"locked_validation_threshold.json").write_text(json.dumps({"threshold":.5}))
            for seed in (17,42,73):
                ck=root/f"seed{seed}.pt"; ck.write_bytes(b"x")
                jobs.append({"experiment_id":f"maxvit512__R__seed{seed}","architecture":"maxvit512","dataset_variant_id":"R","training_policy":"maxvit512_standard","seed":seed,"checkpoint_path":ck.name})
            (root/"configs/classifier_experiment_matrix.json").write_text(json.dumps({"jobs":jobs})); (root/"configs/classifier_training_protocols.json").write_text(json.dumps({"policies":{"maxvit512":{}}}))
            fake=types.SimpleNamespace(verify_lock_still_valid=lambda _:(True,[]),LOCK_DIR="results/final_evaluation_v2")
            with patch.dict(sys.modules,{"finalize_locked_test_stage":fake}): manifest=locked.run_locked(root,lambda job,ck,rows:[.2,.8])
            self.assertEqual(len(manifest["outputs"]),1); self.assertTrue((lock_dir/"LOCKED_TEST_COMPLETED").is_file())
            with patch.dict(sys.modules,{"finalize_locked_test_stage":fake}), self.assertRaises(PermissionError): locked.run_locked(root,lambda *args:[])


if __name__=="__main__": unittest.main()
