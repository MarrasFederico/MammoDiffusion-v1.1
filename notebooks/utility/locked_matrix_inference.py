"""One-shot v2 locked-test inference. Importing this module never opens the test split."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2)+"\n"); tmp.replace(path)


def _panel_ids(lock_dir: Path) -> dict[str, list[str]]:
    result={}
    for name in ("primary", "secondary", "ablation"):
        payload=json.loads((lock_dir/f"{name}_panel_manifest.json").read_text()); result[name]=payload.get("experiment_ids",[])
    return result


def run_locked(root: Path, predictor_fn=None) -> dict:
    """Verify lock first, then infer each three-seed ensemble exactly once.

    ``predictor_fn(job, checkpoint, rows)`` exists solely for a synthetic fixture test; the
    production path loads the registered architecture adapter and local checkpoint.
    """
    root=Path(root); sys.path.insert(0,str(root/"scripts")); sys.path.insert(0,str(root/"notebooks/utility"))
    import finalize_locked_test_stage as lock
    valid,problems=lock.verify_lock_still_valid(root)
    if not valid: raise PermissionError("locked matrix is unavailable: "+"; ".join(problems))
    lock_dir=root/lock.LOCK_DIR; done=lock_dir/"LOCKED_TEST_COMPLETED"
    if done.is_file(): raise PermissionError("locked test is one-shot and has already completed")
    test_csv=root/"data/processed/metadata/test.csv"; test_rows=_rows(test_csv)
    matrix=json.loads((root/"configs/classifier_experiment_matrix.json").read_text()); jobs=matrix["jobs"]
    protocols=json.loads((root/"configs/classifier_training_protocols.json").read_text())["policies"]
    outputs=[]
    for panel,ids in _panel_ids(lock_dir).items():
        for logical in ids:
            selected=[j for j in jobs if j["experiment_id"]==logical or j["experiment_id"].startswith(logical+"__seed")]
            if len(selected)==1:
                stem=selected[0]["experiment_id"].rsplit("__seed",1)[0]
                selected=[j for j in jobs if j["experiment_id"].startswith(stem+"__seed")]
            by_seed={int(j["seed"]):j for j in selected}
            if set(by_seed)!={17,42,73}: raise RuntimeError(f"{logical}: locked ensemble requires seeds 17/42/73")
            seed_probs=[]
            for seed in (17,42,73):
                job=by_seed[seed]; checkpoint=root/job["checkpoint_path"]
                if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
                if predictor_fn: probs=list(map(float,predictor_fn(job,checkpoint,test_rows)))
                else:
                    from classifier_architecture_adapters import get_adapter
                    adapter=get_adapter(job["architecture"],protocols[job["architecture"]],root)
                    probs=adapter.predict_validation(checkpoint,[{**r,"processed_path":r.get("processed_path") or r.get("path")} for r in test_rows],seed=seed)["probabilities"]
                if len(probs)!=len(test_rows): raise RuntimeError("locked prediction length mismatch")
                seed_probs.append(probs)
            base=root/"results/classifiers_matrix"/by_seed[17]["architecture"]/by_seed[17]["dataset_variant_id"]/by_seed[17]["training_policy"]
            threshold_payload=json.loads((base/"ensemble/metrics/locked_validation_threshold.json").read_text()); threshold=float(threshold_payload["threshold"])
            out_rows=[]
            for index,row in enumerate(test_rows):
                ensemble=sum(values[index] for values in seed_probs)/3
                out_rows.append({"patient_id":row["patient_id"],"image_id":row["image_id"],"label":int(row["label"]),
                    "prob_seed_17":seed_probs[0][index],"prob_seed_42":seed_probs[1][index],"prob_seed_73":seed_probs[2][index],
                    "prob_ensemble":ensemble,"predicted_label":int(ensemble>=threshold),"threshold":threshold})
            if len({(r["patient_id"],r["image_id"]) for r in out_rows})!=len(out_rows): raise RuntimeError("duplicate locked patient/image key")
            out=lock_dir/"predictions"/panel/f"{logical}.csv"; out.parent.mkdir(parents=True,exist_ok=True)
            with out.open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,list(out_rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(out_rows)
            outputs.append({"panel":panel,"experiment_id":logical,"path":str(out.relative_to(root)),"sha256":_sha(out)})
    manifest={"schema_version":1,"test_csv_sha256":_sha(test_csv),"outputs":outputs,"one_shot":True}
    _atomic_json(lock_dir/"locked_test_predictions_manifest.json",manifest); _atomic_json(done,{"manifest_sha256":_sha(lock_dir/"locked_test_predictions_manifest.json")})
    return manifest
