import collections,math,os,shutil,subprocess
from pathlib import Path
from protein_inference.io import sha,save

def check_foldseek(binary, spec):
    version = subprocess.check_output([binary, 'version'], text=True).strip()
    expected_version = spec['foldseek_version']
    if expected_version and version != expected_version:
        raise ValueError(f'Foldseek {expected_version} required, found {version} at {binary}')
    expected_hash = spec['foldseek_sha256']
    if expected_hash and sha(binary) != expected_hash:
        raise ValueError(f'Foldseek binary hash mismatch at {binary}')
    return version


def cluster(files,dest,c):
    spec=c['evaluation'];binary=shutil.which(spec['foldseek']) or spec['foldseek']
    version = check_foldseek(binary, spec)
    dest.mkdir(parents=True,exist_ok=False);inp=dest/'pdb';inp.mkdir()
    if not files:return {'n':0,'clusters':0,'effective_clusters':0.,'sizes':[]}
    names=[]
    for i,f in enumerate(files):
        name=f'item-{i:06d}.pdb';shutil.copyfile(f,inp/name);names.append(name)
    if len(files)==1:sizes=[1]
    else:
        cmd=[binary,'easy-cluster',str(inp),str(dest/'res'),str(dest/'tmp'),*map(str,spec['foldseek_arguments'])]
        with (dest/'foldseek.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':''})
        pairs=[x.split('\t') for x in (dest/'res_cluster.tsv').read_text().splitlines()]
        if len(pairs)!=len(files) or len({v[1] for v in pairs})!=len(files):raise ValueError('Incomplete Foldseek assignment')
        sizes=list(collections.Counter(x[0] for x in pairs).values())
    n=sum(sizes);r={'n':n,'clusters':len(sizes),'effective_clusters':math.exp(-sum(v/n*math.log(v/n) for v in sizes)),'sizes':sizes,'foldseek_version':version,'binary_sha256':sha(binary)};save(dest/'clusters.json',r);return r
