"""Safe, recursive YAML configuration; all paths resolve from repository root."""
import copy, hashlib, json, os, re
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
MODELS = ('esm3', 'dplm2', 'dplm21')
TASKS = ('joint', 'motif', 'structure_prediction', 'inverse_folding')
def merge(a, b):
    a = copy.deepcopy(a)
    for k, v in b.items():
        a[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else copy.deepcopy(v)
    return a

def expand(v):
    if isinstance(v, dict): return {k: expand(x) for k,x in v.items()}
    if isinstance(v, list): return [expand(x) for x in v]
    if not isinstance(v, str): return v
    def env(m):
        key, _, default = m.group(1).partition(':-')
        if key not in os.environ and not default: raise ValueError('Set environment variable '+key)
        return os.environ.get(key, default)
    return re.sub(r'\$\{([^}]+)\}', env, v)

def load(path, stack=()):
    p = Path(path).resolve()
    if p in stack: raise ValueError('Cyclic YAML inheritance: '+str(p))
    v = yaml.safe_load(p.read_text())
    if not isinstance(v, dict): raise ValueError('YAML root must be a mapping')
    bases = v.pop('extends', [])
    if isinstance(bases, str): bases = [bases]
    out = {}
    for b in bases: out = merge(out, load(p.parent/b, stack+(p,)))
    return merge(out, v)

def validate(c):
    if c['model']['family'] not in MODELS or c['task'] not in TASKS: raise ValueError('Unknown model/task')
    m=c['memory']; d=c['decoding']; g=c['guidance']
    if d['order']!='synchronous':raise ValueError('This release entry reproduces synchronous paper recipes; sequential reference recipes need their native entrypoint')
    if m['summary'] not in ('topk','full_mean'):raise ValueError('Unknown memory summary')
    if not 0 <= m['lambda'] <= 1: raise ValueError('memory.lambda must be in [0,1]')
    if not isinstance(m['k'], int) or m['k'] < 1: raise ValueError('memory.k must be a positive integer')
    if m['schedule'] not in ('static','static_half','linear_decay','cosine_decay','early_half','late_half'): raise ValueError('Unknown lambda schedule')
    allowed = {'joint': {'sequence','structure'}, 'motif': {'sequence','structure'}, 'structure_prediction': {'structure'}, 'inverse_folding': {'sequence'}}[c['task']]
    if not set(m['tracks']) <= allowed: raise ValueError('Cannot apply PM to a fixed condition track')
    if c['task'] in ('structure_prediction','inverse_folding') and g['enabled']: raise ValueError('Paper single-modal recipes have CFG disabled')
    if g['enabled'] and (g['sequence'] <= 0 or g['structure'] <= 0): raise ValueError('CFG strengths must be positive; disabled CFG uses its own branch path')
    if d['steps'] not in ('length','length/8') and (not isinstance(d['steps'], int) or d['steps'] < 1): raise ValueError('Invalid decoding steps')
    if c['sampling']['per_length'] < 1 or c['sampling']['batch_size'] < 1: raise ValueError('Invalid sample budget')
    seeds = c['sampling'].get('seeds')
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError('sampling.seeds must be a nonempty list of unsigned 32-bit integers')
    if len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate seeds')
    if c['model']['family'] != 'esm3' and m['interface'] != 'post_ln': raise ValueError('This release uses the validated post-LN DPLM interface')
    if c['model']['family'] == 'esm3' and m['interface'] != 'token_embedding': raise ValueError('ESM3 reads token embeddings')
    return c

def resolved(path, overrides=()):
    c=load(path)
    for x in overrides:
        key, sep, raw=x.partition('=')
        if not sep: raise ValueError('Override must be path.to.key=value')
        cur=c; keys=key.split('.')
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k],dict): raise ValueError('Unknown override '+key)
            cur=cur[k]
        if keys[-1] not in cur: raise ValueError('Unknown override '+key)
        cur[keys[-1]]=yaml.safe_load(raw)
    return validate(expand(c))

def path(v):
    p=Path(v).expanduser()
    return p if p.is_absolute() else ROOT/p

def digest(c): return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
