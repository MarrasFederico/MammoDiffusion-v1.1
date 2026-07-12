#!/usr/bin/env python3
"""Selective, non-overwriting migration of the four proven legacy experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "Versione vecchia" / "experiments"

MAPPINGS = {
 "01_sd21_baseline_50steps": {"legacy":"20260607_sd21_rsna_mlo_512","notebook":"01_SD21_Baseline_50steps.ipynb","selected":"checkpoint-3000","files":[
   ("model/checkpoint-3000/unet","model/checkpoint-3000/unet"),("model/checkpoint-8000/unet","model/checkpoint-8000/unet"),
   ("checkpoint_validation_metrics.json","checkpoint_validation_metrics.json"),("checkpoint_validation_cache.json","checkpoint_validation_cache.json"),
   ("generation_info.json","generation_info.json"),("final_test_metrics.json","final_test_metrics.json"),("final_test_metrics.csv","final_test_metrics.csv"),
   ("sustainability_finetuning.jsonl","sustainability_finetuning.jsonl"),("sustainability_validation.jsonl","sustainability_validation.jsonl"),("sustainability_generation.jsonl","sustainability_generation.jsonl"),
   ("generated_images/final","generated_images/final")],"image_sets":[("generated_images/final/negative","gen_",1361),("generated_images/final/positive","gen_",1361)]},
 "02_sd21_filtered_100steps": {"legacy":"20260611_sd21_rsna_mlo_512_inference_100_steps","notebook":"02_SD21_Filtered_100steps.ipynb","selected":"checkpoint-3000","files":[
   ("model/checkpoint-3000/unet","model/checkpoint-3000/unet"),("model/checkpoint-8000/unet","model/checkpoint-8000/unet"),
   ("generated_images/final","generated_images/final"),("generated_images/raw_matched_1361","generated_images/raw_matched_1361")],
   "image_sets":[("generated_images/final/negative","gen_",2722),("generated_images/final/positive","gen_",2722),("generated_images/raw_matched_1361/negative","gen_",1361),("generated_images/raw_matched_1361/positive","gen_",1361)]},
 "05_ldm_basic_fromscratch": {"legacy":"20260617_ldm_basic","notebook":"05_LDM_Basic_FromScratch.ipynb","selected":"step_70000","files":[
   ("checkpoints_ldm/ldm_unet_best_eval.keras","checkpoints_ldm/ldm_unet_best_eval.keras"),("checkpoints_ldm/ldm_unet_final_step080000.keras","checkpoints_ldm/ldm_unet_final_step080000.keras"),
   ("models/vae_encoder_best.keras","models/vae_encoder_best.keras"),("models/vae_decoder_best.keras","models/vae_decoder_best.keras"),
   ("latents","latents"),("logs","logs"),("evaluation","evaluation"),("synthetic_raw","synthetic_raw"),("synthetic_filtered","synthetic_filtered")],
   "image_sets":[("synthetic_raw","synth_",2722),("synthetic_filtered","synth_filtered_",1361)]},
 "06_ldm_extra1361_fromscratch": {"legacy":"20260619_ldm_extra1361","notebook":"06_LDM_Extra1361_FromScratch.ipynb","selected":"step_70000","files":[
   ("checkpoints_ldm/ldm_unet_best_eval.keras","checkpoints_ldm/ldm_unet_best_eval.keras"),("checkpoints_ldm/ldm_unet_final_step080000.keras","checkpoints_ldm/ldm_unet_final_step080000.keras"),
   ("models/vae_encoder_best.keras","models/vae_encoder_best.keras"),("models/vae_decoder_best.keras","models/vae_decoder_best.keras"),
   ("latents","latents"),("logs","logs"),("evaluation","evaluation"),("synthetic_raw","synthetic_raw"),("synthetic_raw_negative","synthetic_raw_negative"),("synthetic_filtered","synthetic_filtered")],
   "image_sets":[("synthetic_raw","synth_",4083),("synthetic_raw_negative","synth_",4083),("synthetic_filtered","synth_filtered_",1361)]},
}

def digest(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 return h.hexdigest()

def copy_item(src: Path, dst: Path, dry: bool) -> list[dict]:
 files=[]
 sources=[src] if src.is_file() else sorted(p for p in src.rglob('*') if p.is_file() and not p.is_symlink())
 for item in sources:
  rel=Path(item.name) if src.is_file() else item.relative_to(src); target=dst if src.is_file() else dst/rel
  sha=digest(item)
  if target.exists():
   if not target.is_file() or target.stat().st_size!=item.stat().st_size or digest(target)!=sha: raise FileExistsError(f"collision: {target}")
  elif not dry:
   target.parent.mkdir(parents=True,exist_ok=True)
   try: os.system(f"cp --reflink=auto --preserve=mode,timestamps -- {shlex_quote(str(item))} {shlex_quote(str(target))}") or None
   except Exception: shutil.copy2(item,target)
   if not target.is_file() or digest(target)!=sha: raise IOError(f"post-copy signature mismatch: {target}")
  files.append({"legacy_path":str(item),"canonical_path":str(target),"size_bytes":item.stat().st_size,"sha256_before":sha,"sha256_after":sha,"action":"existing_identical" if target.exists() and dry else "copied_or_verified"})
 return files

def shlex_quote(value: str) -> str:
 import shlex; return shlex.quote(value)

def file_spec(root: Path, relative: str) -> dict:
 p=root/relative; return {"path":relative,"size_bytes":p.stat().st_size,"sha256":digest(p)}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args(); report=[]
 for exp_id,cfg in MAPPINGS.items():
  src=OLD/cfg['legacy']; dst=ROOT/'experiments/diffusers'/exp_id; copied=[]
  if not args.dry_run: dst.mkdir(parents=True,exist_ok=True)
  for old_rel,new_rel in cfg['files']: copied += copy_item(src/old_rel,dst/new_rel,args.dry_run)
  runtime={"schema_version":1,"experiment_id":exp_id,"legacy_experiment":cfg['legacy'],"provenance":"legacy_normalized_verified","selected_checkpoint":cfg['selected'],"phases":{}}
  if not args.dry_run:
   checkpoint = "model/checkpoint-8000/unet/diffusion_pytorch_model.safetensors" if exp_id.startswith(('01','02')) else "checkpoints_ldm/ldm_unet_final_step080000.keras"
   best = "model/checkpoint-3000/unet/diffusion_pytorch_model.safetensors" if exp_id.startswith(('01','02')) else "checkpoints_ldm/ldm_unet_best_eval.keras"
   runtime['phases']['training']={"files":[file_spec(dst,checkpoint)]}
   runtime['phases']['checkpoint_selection']={"files":[file_spec(dst,best)]}
   generation_sets=[]
   for p,prefix,count in cfg['image_sets']:
    if 'filtered' in p: continue
    if exp_id.startswith(('01','02')): generation_sets.append({"path":p,"allowed_prefixes":["gen_","eval_neg_","eval_pos_"],"count":count})
    else: generation_sets.append({"path":p,"prefix":prefix,"start_index":0,"count":count})
   runtime['phases']['generation']={"image_sets":generation_sets}
   filtered=[{"path":p,"prefix":prefix,"start_index":0,"count":count} for p,prefix,count in cfg['image_sets'] if 'filtered' in p]
   runtime['phases']['filter']={"image_sets":filtered} if filtered else {"files":[file_spec(dst,best)]}
   if exp_id == '01_sd21_baseline_50steps': evaluation_rel='checkpoint_validation_metrics.json'
   elif exp_id == '02_sd21_filtered_100steps': evaluation_rel='../../../results/diffusers/02_sd21_filtered_100steps/metrics/checkpoint_validation_metrics.json'
   else: evaluation_rel='evaluation/checkpoint_metrics.json'
   runtime['phases']['evaluation']={"files":[file_spec(dst,evaluation_rel)]}
   (dst/'legacy_runtime_manifest.json').write_text(json.dumps(runtime,indent=2)+"\n")
  report.append({"legacy_experiment_path":str(src),"canonical_experiment_id":exp_id,"canonical_experiment_path":str(dst),"legacy_notebook":"notebooks/2_diffusers/legacy naming inferred from experiment","current_notebook":f"notebooks/2_diffusers/{cfg['notebook']}","configuration_compared":{"selected_checkpoint":cfg['selected'],"compatibility":"compatible_with_documented_legacy_lineage"},"files":copied,"excluded":["diffusers_repo","pretrained_model/stable-diffusion-2-1-base","hf_cache","optimizer.bin","temporary files"],"compatibility_status":"promoted_after_content_validation","provenance":"legacy_normalized_verified"})
 if not args.dry_run: (ROOT/'configs/legacy_experiment_migration.json').write_text(json.dumps({"schema_version":1,"mappings":report},indent=2)+"\n")
 print(json.dumps({"dry_run":args.dry_run,"experiments":len(report),"files":sum(len(x['files']) for x in report)},indent=2))
if __name__=='__main__': main()
