"""Explicitly download an official fastMRI leaderboard checkpoint, outside inference."""
import argparse
from pathlib import Path
import shutil
import urllib.request


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anatomy',choices=['knee','brain'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    url=f'https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/{args.anatomy}_leaderboard_state_dict.pt'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(url,timeout=60) as response, args.output.open('wb') as target:
        shutil.copyfileobj(response,target)
    print(url)
    print('Architecture: num_cascades=12, chans=18, pools=4, sens_chans=8, sens_pools=4')
