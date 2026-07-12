#!/usr/bin/env python3
"""Create local content-aware runtime manifests for completed experiments 03/04/07/08."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPECS={
 '03_sd21_vae_finetuned':{'checkpoint':'model/checkpoint-4000/unet/diffusion_pytorch_model.safetensors','final':'model/checkpoint-8000/unet/diffusion_pytorch_model.safetensors','metrics':'../../../results/diffusers/03_sd21_vae_finetuned/metrics/checkpoint_validation_metrics.json','raw':('generated_images/final',2722,['gen_','eval_neg_','eval_pos_']),'filtered':'../../../data/synthetic/fine_tuned_vaeft'},
 '04_sd21_lora':{'checkpoint':'model/checkpoint-4500/pytorch_lora_weights.safetensors','final':'model/pytorch_lora_weights.safetensors','metrics':'../../../results/diffusers/04_sd21_lora/metrics/checkpoint_validation_metrics.json','raw':('generated_images/final',2722,['gen_','eval_neg_','eval_pos_']),'filtered':'../../../data/synthetic/fine_tuned_lora'},
 '07_ldm_sdvae_extra1361':{'checkpoint':'checkpoints_ldm/ldm_unet_best_eval.keras','final':'checkpoints_ldm/ldm_unet_final_step150000.keras','metrics':'../../../results/diffusers/07_ldm_sdvae_extra1361/metrics/checkpoint_metrics.json','raw':('synthetic_raw',4083,['synth_']),'raw_negative':('synthetic_raw_negative',4083,['synth_']),'filtered':'../../../data/synthetic/fromscratch_new'},
 '08_ldm_v3_sdvae_fromscratch':{'checkpoint':'checkpoints_ldm/ldm_unet_best_eval.keras','final':'checkpoints_ldm/ldm_unet_final_step150000.keras','metrics':'../../../results/diffusers/08_ldm_v3_sdvae_fromscratch/metrics/checkpoint_metrics.json','raw':('synthetic_raw',4083,['synth_']),'raw_negative':('synthetic_raw_negative',4083,['synth_']),'filtered':'../../../data/synthetic/fromscratch_v3'},
}
def sig(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 return {'path':path.name,'size_bytes':path.stat().st_size,'sha256':h.hexdigest()}
def main():
 for name,s in SPECS.items():
  base=ROOT/'experiments/diffusers'/name
  def fs(rel):
   p=base/rel; x=sig(p); x['path']=rel; return x
  raw=[]
  for key in ('raw','raw_negative'):
   if key in s:
    path,count,prefixes=s[key]; raw.append({'path':path+'/negative' if (base/path/'negative').is_dir() else path,'allowed_prefixes':prefixes,'count':count})
    if (base/path/'positive').is_dir(): raw.append({'path':path+'/positive','allowed_prefixes':prefixes,'count':count})
  filtered=[]
  for cls in ('negative','positive'):
   rel=s['filtered']+'/'+cls; filtered.append({'path':rel,'allowed_prefixes':['selected_negative_','selected_pos_','synth_filtered_'],'count':1361})
  payload={'schema_version':1,'experiment_id':name,'provenance':'legacy_normalized_verified','phases':{
   'training':{'files':[fs(s['final'])]},'checkpoint_selection':{'files':[fs(s['checkpoint'])]},
   'generation':{'image_sets':raw},'filter':{'image_sets':filtered},'evaluation':{'files':[fs(s['metrics'])]}}}
  (base/'legacy_runtime_manifest.json').write_text(json.dumps(payload,indent=2)+'\n'); print(name)
if __name__=='__main__': main()
