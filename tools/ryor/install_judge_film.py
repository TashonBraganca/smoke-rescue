"""Install a delivered house recording without changing the player.

Example: python3 tools/ryor/install_judge_film.py family_ranch /path/video2.mp4
Keep result values null until a matching Unity result supplies them. A generic
cinematic and the retired one-door recording are not replacement-house evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('house',choices=['family_ranch','long_ranch','onedoor_934'])
    p.add_argument('source',type=Path)
    p.add_argument('--site',type=Path,default=Path('/Volumes/Tashon/opencv/showcase/site-next'))
    p.add_argument('--people-found',type=int)
    p.add_argument('--people-total',type=int,default=5)
    p.add_argument('--mission-seconds',type=float)
    p.add_argument('--both-out',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--evidence',help='Matching Unity result or recording notes for the reported metrics')
    args=p.parse_args()
    if any(v is not None for v in [args.people_found,args.mission_seconds,args.both_out]) and not args.evidence:
        p.error('Result values need --evidence pointing to the matching Unity result.')
    if args.people_found is not None and not 0<=args.people_found<=args.people_total:p.error('Invalid people count')
    if args.mission_seconds is not None and args.mission_seconds<0:p.error('Invalid mission time')
    dest=args.site/'public/film/judge'/(args.house+'.mp4');dest.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(args.source),'-vf','scale=1280:-2','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-c:a','aac','-b:a','96k','-movflags','+faststart',str(dest)],check=True)
    config=args.site/'public/rescue-data/missions.json';c=json.loads(config.read_text())
    c['films'][args.house]=dict(src='/film/judge/'+dest.name,peopleFound=args.people_found,peopleTotal=args.people_total,missionSeconds=args.mission_seconds,allOut=args.both_out,note='Recorded Unity mission',source_file=args.source.name,source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),evidence=args.evidence)
    config.write_text(json.dumps(c,indent=2)+'\n')
    print('Installed',args.house,dest.stat().st_size,'bytes')

if __name__=='__main__':main()
