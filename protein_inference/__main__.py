import argparse, json, os, sys, traceback
from pathlib import Path
from protein_inference.config import ROOT, resolved, path, digest

def main():
    p=argparse.ArgumentParser(description='Multimodal protein inference')
    p.add_argument('command',choices=['validate','run','evaluate','analyze','doctor'])
    p.add_argument('--config',required=True); p.add_argument('--set',action='append',default=[])
    a=p.parse_args(); c=resolved(a.config,a.set)
    if a.command=='validate': print(json.dumps(c,indent=2)); return
    if a.command=='doctor':
        import importlib.util, platform, shutil
        print(json.dumps({'python':platform.python_version(),'model':c['model'],'yaml':True,'torch_installed':bool(importlib.util.find_spec('torch')),'foldseek':shutil.which(c['evaluation']['foldseek']),'esm_source':str(ROOT/'upstream/esm'),'dplm_source':str(ROOT/'upstream/dplm'),'runtime_kernel_tested':False},indent=2));return
    for k,v in c['runtime']['environment'].items(): os.environ[k]=str(path(v)) if k in ('HF_HOME','TORCH_HOME') else str(v)
    sys.path[:0]=([str(ROOT/'upstream/esm')] if c['model']['family']=='esm3' else [])+[str(ROOT/'upstream/dplm/src'),str(ROOT/'upstream/dplm/vendor/openfold'),str(ROOT/'upstream/dplm'),str(ROOT/'upstream/evals')]
    if a.command=='analyze':
        from protein_inference.analysis import run
        run(c);return
    if a.command=='evaluate':
        from protein_inference.evaluation import run
        run(c);return
    from protein_inference.io import save, source_manifest
    backend=__import__('protein_inference.backends.'+('esm3' if c['model']['family']=='esm3' else 'dplm'),fromlist=['run'])
    for seed in c['sampling']['seeds']:
        out=(path(c['output'])/f'seed-{seed}').resolve()
        out.mkdir(parents=True,exist_ok=False)
        save(out/'resolved-config.json',c);save(out/'source-manifest.json',source_manifest())
        save(out/'STATUS.json',{'stage':'started','pid':os.getpid(),'config_sha256':digest(c),'seed':seed})
        try:
            backend.run(c,seed,out)
            save(out/'GENERATION-COMPLETE.json',{'passed':True,'seed':seed,'config_sha256':digest(c)})
            if c['evaluation']['after_generation']:
                from protein_inference.evaluation import evaluate_seed
                evaluate_seed(c,out)
            save(out/'STATUS.json',{'stage':'complete','seed':seed,'config_sha256':digest(c)})
        except BaseException:
            save(out/'FAILED.json',{'error':traceback.format_exc(),'seed':seed});raise
if __name__=='__main__': main()
