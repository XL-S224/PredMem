import hashlib,json,re
from pathlib import Path
from PredMem.config import ROOT

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.pending');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def key(*v): return int(hashlib.sha256('|'.join(map(str,v)).encode()).hexdigest()[:8],16)%2147483647
def safe_name(s):
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+',s): raise ValueError('Unsafe input identifier '+s)
    return s

def fasta(p):
    out={}; k=None
    for l in Path(p).read_text().splitlines():
        if l.startswith('>'):
            k=safe_name(l[1:].split()[0])
            if k in out: raise ValueError('Duplicate FASTA identifier '+k)
            out[k]=''
        elif l.strip():
            if k is None: raise ValueError('Sequence before FASTA header')
            out[k]+=l.strip()
    if not out: raise ValueError('Empty FASTA')
    return out

def source_manifest():
    files=list((ROOT/'PredMem').rglob('*.py'))+list((ROOT/'configs').rglob('*.yaml'))+list((ROOT/'scripts').rglob('*.py'))
    files+=list((ROOT/'upstream/dplm/src').rglob('*.py'))+list((ROOT/'upstream/esm/esm').rglob('*.py'))+list((ROOT/'upstream/evals').glob('*.py'))
    files += [ROOT/'upstream/esm/esm/sdk/experimental/beam_search.py',ROOT/'upstream/dplm/src/byprot/models/dplm2/dplm2.py',ROOT/'upstream/dplm/src/byprot/models/dplm2/dplm2_bit.py']
    files += [ROOT/'upstream/dplm/guided_generate_dplm2.py',ROOT/'upstream/dplm/guided_scaffold_generate_dplm2.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
