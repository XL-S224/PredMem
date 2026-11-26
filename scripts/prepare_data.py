"""Extract trusted benchmark archives with path/link/overwrite checks."""
import argparse,hashlib,json,tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--destination',default='data');a=p.parse_args();archive=Path(a.archive);dest=Path(a.destination).resolve();dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:*') as tar:
        members=tar.getmembers()
        for m in members:
            target=(dest/m.name).resolve()
            if not target.is_relative_to(dest) or m.issym() or m.islnk() or not (m.isfile() or m.isdir()):raise ValueError('Unsafe archive member '+m.name)
            if m.isfile() and target.exists():raise FileExistsError(target)
        for m in members:
            target=dest/m.name
            if m.isdir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(m) as src,target.open('xb') as out:
                    __import__('shutil').copyfileobj(src,out)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest();(dest/(archive.name+'.receipt.json')).write_text(json.dumps({'archive_name':archive.name,'archive_sha256':digest,'members':len(members)},indent=2));print('EXTRACTED',archive.name)
if __name__=='__main__':main()
