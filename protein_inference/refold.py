"""Independent ESMFold evaluator; run in the evaluation environment."""
import argparse,json,sys
from pathlib import Path
from protein_inference.config import ROOT
from protein_inference.io import save

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
    sys.path.insert(0,str(ROOT/'upstream/dplm/vendor/openfold'))
    import torch
    from transformers import AutoTokenizer,EsmForProteinFolding
    records=json.loads(Path(a.manifest).read_text());tokenizer=AutoTokenizer.from_pretrained(a.checkpoint)
    model=EsmForProteinFolding.from_pretrained(a.checkpoint,low_cpu_mem_usage=True).cuda().eval()
    model.trunk.set_chunk_size(128)
    out=Path(a.manifest).parent/'refolded';out.mkdir(exist_ok=True);rows=[]
    for r in records:
        dest=out/r['group']/(r['id']+'.pdb');dest.parent.mkdir(parents=True,exist_ok=True)
        receipt=dest.with_suffix('.json')
        if receipt.exists() and dest.exists(): rows.append(json.loads(receipt.read_text()));continue
        ids=tokenizer([r['sequence']],return_tensors='pt',add_special_tokens=False)['input_ids'].cuda()
        with torch.inference_mode(): z=model(ids)
        raw=float(z['plddt'][z['atom37_atom_exists'].bool()].mean())
        cpu={k:v.cpu() for k,v in z.items()};dest.write_text(model.output_to_pdb(cpu)[0])
        row={'group':r['group'],'id':r['id'],'pdb':str(dest),'plddt':100*raw,'plddt_raw_0_to_1':raw}
        if not 0 <= raw <= 1.001: raise ValueError('Unexpected ESMFold confidence scale; do not silently rescale')
        save(receipt,row);rows.append(row);del z,cpu,ids
    save(out/'index.json',rows)
if __name__=='__main__':main()
