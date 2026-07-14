"""One-shot v2 locked-test inference. Importing this module never opens the test split."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classifier_pipeline_contracts import (  # noqa: E402
    PIPELINE_NAMESPACE, atomic_json, code_revision, signed_payload, verify_signed_payload,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def _atomic_json(path: Path, payload) -> None:
    atomic_json(path, payload)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _panel_ids(lock_dir: Path) -> dict[str, list[str]]:
    result={}
    for name in ("primary", "secondary", "ablation"):
        payload=json.loads((lock_dir/f"{name}_panel_manifest.json").read_text()); result[name]=payload.get("experiment_ids",[])
    return result


def run_locked(root: Path, predictor_fn=None, incident_token: str | None = None) -> dict:
    """Verify lock first, then infer each three-seed ensemble exactly once.

    ``predictor_fn(job, checkpoint, rows)`` exists solely for a synthetic fixture test; the
    production path loads the registered architecture adapter and local checkpoint.
    """
    root=Path(root); sys.path.insert(0,str(root/"scripts")); sys.path.insert(0,str(root/"notebooks/utility"))
    import finalize_locked_test_stage as lock
    lock_dir=root/lock.LOCK_DIR; started=lock_dir/"LOCKED_TEST_STARTED"; done=lock_dir/"LOCKED_TEST_COMPLETE"
    failed=lock_dir/"LOCKED_TEST_FAILED"; authorization=lock_dir/"LOCKED_TEST_RETRY_AUTHORIZATION"
    if done.is_file() or (lock_dir/"LOCKED_TEST_COMPLETED").is_file():
        raise PermissionError("locked test is one-shot: valid outputs already exist")
    lock_path=lock_dir/"EXPERIMENT_MATRIX_LOCKED"
    if not lock_path.is_file():
        if predictor_fn is None:
            raise PermissionError("locked matrix is unavailable: no scientific lock marker")
        # Dependency-injected fixture compatibility only. Production inference has no
        # predictor_fn and can never use this branch.
        valid,problems=lock.verify_lock_still_valid(root)
        if not valid: raise PermissionError("locked matrix is unavailable: "+"; ".join(problems))
        lock_payload={"lock_signature":"dependency-injected-fixture"}
    else:
        lock_payload=json.loads(lock_path.read_text())
    if lock_payload.get("pipeline_namespace") is not None:
        verify_signed_payload(lock_payload)
    if started.is_file():
        if not (incident_token and failed.is_file() and authorization.is_file()):
            raise PermissionError("prior locked attempt requires a signed technical-retry authorization")
        auth=json.loads(authorization.read_text()); verify_signed_payload(auth)
        if auth.get("artifact_type") != "classifier_locked_retry_authorization" or \
           auth.get("incident_id") != incident_token or auth.get("lock_signature") != lock_payload.get("lock_signature"):
            raise PermissionError("technical-retry authorization does not match this lock/incident")
    else:
        marker=signed_payload({"schema_version":2,"pipeline_namespace":PIPELINE_NAMESPACE,
            "artifact_type":"classifier_locked_inference_start","status":"started","attempt":1,
            "lock_signature":lock_payload.get("lock_signature"),"code_revision":code_revision(root),
            "environment":{"python":sys.version.split()[0],"platform":platform.platform()},
            "started_at_unix":time.time(),"policy":"crash_requires_signed_incident_review"})
        started.parent.mkdir(parents=True,exist_ok=True)
        try:
            fd=os.open(started,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError as exc:
            raise PermissionError("another locked inference process already started") from exc
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            stream.write(json.dumps(marker,indent=1)+"\n"); stream.flush(); os.fsync(stream.fileno())
    try:
        # The durable start marker is created before this full verification, because the full
        # verification intentionally hashes the locked CSV and referenced images.
        valid,problems=lock.verify_lock_still_valid(root)
        if not valid: raise PermissionError("locked matrix is unavailable: "+"; ".join(problems))
        test_csv=root/"data/processed/metadata/test.csv"; test_rows=_rows(test_csv)
    # processed_path in test.csv may be relative; resolve it against the project root
    # explicitly so the locked dataloader never depends on the caller's current working
    # directory (a notebook launched from a different cwd would otherwise silently miss files).
        for row in test_rows:
            rel = row.get("processed_path")
            if rel and not Path(rel).is_absolute():
                row["processed_path"] = str((root / rel).resolve())
        matrix=json.loads((root/"configs/classifier_experiment_matrix.json").read_text()); jobs=matrix["jobs"]
        protocols=json.loads((root/"configs/classifier_training_protocols.json").read_text())["policies"]
        outputs=[]
        for panel,ids in _panel_ids(lock_dir).items():
            seen_stems=set()
            for logical in ids:
            # Accept either the canonical logical ensemble id ("arch__variant__ensemble") or,
            # defensively, a bare stem/individual seed id - always resolved to one *logical
            # stem* per configuration, so a panel that (still) lists the same configuration
            # more than once under different names infers and writes it only once.
                if logical.endswith("__ensemble"):
                    stem=logical[:-len("__ensemble")]
                elif "__seed" in logical:
                    stem=logical.rsplit("__seed",1)[0]
                else:
                    stem=logical
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)
                selected=[j for j in jobs if j["experiment_id"].startswith(stem+"__seed")]
                by_seed={int(j["seed"]):j for j in selected}
                canonical=stem+"__ensemble"
                if set(by_seed)!={17,42,73}: raise RuntimeError(f"{canonical}: locked ensemble requires seeds 17/42/73")
                out=lock_dir/"predictions"/panel/f"{canonical}.csv"
                if out.is_file():
                    # A reviewed retry reuses a fully atomic prior table; it never evaluates
                    # the same finalist twice or overwrites observed locked predictions.
                    existing_rows=_rows(out)
                    if len(existing_rows)!=len(test_rows):
                        raise RuntimeError(f"partial/incompatible locked output requires manual quarantine: {out}")
                    outputs.append({"panel":panel,"experiment_id":canonical,"path":str(out.relative_to(root)),"sha256":_sha(out),"reused":True})
                    continue
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
                _atomic_csv(out,out_rows)
                outputs.append({"panel":panel,"experiment_id":canonical,"path":str(out.relative_to(root)),"sha256":_sha(out),"reused":False})
        manifest=signed_payload({"schema_version":2,"pipeline_namespace":PIPELINE_NAMESPACE,
            "artifact_type":"classifier_locked_predictions","lock_signature":lock_payload.get("lock_signature"),
            "code_revision":code_revision(root),"test_csv_sha256":_sha(test_csv),"outputs":outputs,"one_shot":True,
            "start_marker_signature":json.loads(started.read_text()).get("signature"),
            "environment":json.loads(started.read_text()).get("environment"),
            "technical_retry_incident":incident_token})
        _atomic_json(lock_dir/"locked_test_predictions_manifest.json",manifest)
        _atomic_json(done,signed_payload({"schema_version":2,"pipeline_namespace":PIPELINE_NAMESPACE,
            "artifact_type":"classifier_locked_inference_completion","manifest_sha256":_sha(lock_dir/"locked_test_predictions_manifest.json"),
            "lock_signature":lock_payload.get("lock_signature")}))
        return manifest
    except BaseException as exc:
        _atomic_json(failed,signed_payload({"schema_version":2,"pipeline_namespace":PIPELINE_NAMESPACE,
            "artifact_type":"classifier_locked_inference_failure","lock_signature":lock_payload.get("lock_signature"),
            "error_type":type(exc).__name__,"message":str(exc),"failed_at_unix":time.time()}))
        raise
