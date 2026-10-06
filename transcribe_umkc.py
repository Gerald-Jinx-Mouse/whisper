#!/usr/bin/env python3
"""Resume-safe local UMKC transcription with the installed Whisper project.

Uses turbo on CUDA; preserves existing transcripts. Writes txt/srt/json and a
timestamped Markdown transcript beside each recording. Study notes are reviewed
separately. Silent Class_Video recordings share the same-folder Class_Audio track.
"""
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from difflib import SequenceMatcher

import torch
import whisper
from whisper.utils import get_writer
from transcripts_to_md import render

ROOT=Path('/home/jerry/Projects/UMKC')
LOG=Path(__file__).parent/'logs/umkc-transcription'
LOG.mkdir(parents=True,exist_ok=True)
PROBE=json.loads((LOG/'video-probe.json').read_text())
STATE=LOG/'status.json'
rows=[]
for p in PROBE:
 v=Path(p['video']);audio=any(s.get('codec_type')=='audio' for s in p['probe'].get('streams',[]))
 rows.append(dict(video=str(v),duration_seconds=float(p['probe'].get('format',{}).get('duration',0)),has_audio=audio,status='pending',transcript=None,study_notes=str(v.with_name(v.stem+'_STUDY_NOTES.md'))))
started=time.time()
def save():
 temp=STATE.with_suffix('.json.tmp')
 temp.write_text(json.dumps(dict(started_at_epoch=started,updated_at_epoch=time.time(),model='turbo',device='cuda',records=rows),ensure_ascii=False,indent=2)+'\n')
 temp.replace(STATE)
def log(message):print(time.strftime('%H:%M:%S'),message,flush=True)
def nonempty(p):return p.is_file() and p.stat().st_size>0
def store_result(result,video):
 # All writes are staged, so interruption cannot leave a truncated final output.
 stage=LOG/('stage-'+str(rows.index(next(r for r in rows if r['video']==str(video)))))
 stage.mkdir(exist_ok=True)
 for fmt in ('txt','srt','json'):
  get_writer(fmt,str(stage))(result,str(video),{})
  output=video.with_suffix('.'+fmt);generated=stage/output.name
  if not output.exists():shutil.copyfile(generated,output)
 if not video.with_suffix('.md').exists():
  body=render(video.with_suffix('.srt'),'turbo')
  if body:video.with_suffix('.md').write_text(body)

for r in rows:
 v=Path(r['video'])
 if nonempty(v.with_suffix('.txt')):
  r.update(status='existing_transcript',transcript=str(v.with_suffix('.txt')))
 elif not r['has_audio']:
  paired=v.with_name('Class_Audio.mp4') if v.name=='Class_Video.mp4' else None
  match=next((x for x in rows if paired and x['video']==str(paired) and x['has_audio'] and abs(x['duration_seconds']-r['duration_seconds'])<0.2),None)
  if match:r.update(status='awaiting_paired_audio',audio_source=str(paired))
  else:r.update(status='no_audio',error='No audio stream or matching same-folder Class_Audio recording.')
save()
log(f"Found {len(rows)} recordings; {sum(r['status']=='existing_transcript' for r in rows)} already have transcripts; {sum(r['status']=='awaiting_paired_audio' for r in rows)} silent companion recordings.")
torch.set_num_threads(4)
log('Loading local turbo model on CUDA')
model=whisper.load_model('turbo',device='cuda')
log('Model ready')

# Verify that the untimed Week 4 classroom captions belong to its recording,
# then preserve and reuse that existing transcript rather than transcribing it again.
week4=ROOT/'Fall26/FIN_5553/Week_4_(Sep_14-20)'
caption=week4/"John_Clark's_Finance_Classroom_Captions_English_(United_States).txt"
video=week4/'mp4/Class_Audio.mp4'
row=next((r for r in rows if r['video']==str(video)),None)
if row and row['status']=='pending' and nonempty(caption):
 try:
  log('Checking a 90-second sample against the existing Week 4 captions')
  audio=whisper.load_audio(str(video))[:90*whisper.audio.SAMPLE_RATE]
  sample=model.transcribe(audio,language='en',verbose=None)
  (LOG/'week4-caption-alignment-sample.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
  import re
  clean=lambda s:' '.join(re.findall(r'[a-z0-9]+',s.lower()))
  reference=caption.read_text().split('\n',1)[1]
  reference=re.sub(r'(?m)^John Clark:\s*','',reference)
  words=clean(sample['text']).split();reference_words=clean(reference).split()[:len(words)+60]
  score=SequenceMatcher(None,words,reference_words,autojunk=False).ratio()
  row['caption_sample_similarity']=score
  if score>=0.65:
   shutil.copyfile(caption,video.with_suffix('.txt'))
   md=video.with_suffix('.md')
   if not md.exists():md.write_text('# Class Audio: existing classroom transcript\n\nSource recording: ['+video.name+'](<'+str(video)+'>).\n\nExisting transcript: [original classroom captions](<'+str(caption)+'>). A 90-second Whisper sample confirmed the matching opening. These captions do not contain timestamps.\n\n'+caption.read_text())
   row.update(status='reused_existing_captions',transcript=str(video.with_suffix('.txt')),original_transcript=str(caption),timestamps_available=False)
   log(f'Reused existing Week 4 captions (opening similarity {score:.2f})')
  else:log(f'Caption opening similarity {score:.2f}; will transcribe full recording instead')
 except Exception:
  row['caption_validation_error']=traceback.format_exc();log('Caption validation failed; proceeding with full transcription')
 save()

pending=[r for r in rows if r['status']=='pending']
# Short lectures finish first, making their notes available while longer classes run.
pending.sort(key=lambda r:r['duration_seconds'])
for i,r in enumerate(pending,1):
 v=Path(r['video']);start=time.time();r.update(status='transcribing',started_at_epoch=start);save()
 log(f"[{i}/{len(pending)}] START {v}")
 try:
  result=model.transcribe(str(v),language='en',verbose=None)
  if not result.get('text','').strip():raise ValueError('Whisper returned an empty transcript')
  store_result(result,v)
  r.update(status='transcribed',transcript=str(v.with_suffix('.txt')),elapsed_seconds=round(time.time()-start,2),segments=len(result['segments']))
  log(f"[{i}/{len(pending)}] DONE {v.name} ({r['elapsed_seconds']} seconds)")
 except Exception:
  r.update(status='failed',error=traceback.format_exc());log(f'FAILED {v}: {r["error"]}')
 save()

for r in rows:
 if r['status']!='awaiting_paired_audio':continue
 v=Path(r['video']);source=Path(r['audio_source']);original=next(x for x in rows if x['video']==str(source))
 if not nonempty(source.with_suffix('.txt')):
  r.update(status='failed',error='Paired audio transcript is not available');continue
 for extension in ('.txt','.srt','.json'):
  src=source.with_suffix(extension);dst=v.with_suffix(extension)
  if nonempty(src) and not dst.exists():shutil.copyfile(src,dst)
 if not v.with_suffix('.md').exists():
  source_md=source.with_suffix('.md')
  body=source_md.read_text() if source_md.exists() else source.with_suffix('.txt').read_text()
  v.with_suffix('.md').write_text('# Silent screen recording: transcript from companion audio\n\nThis video has no audio track. Transcript text comes from its same-folder, same-length [Class_Audio recording](<'+str(source)+'>). Any timestamps refer to that audio recording.\n\n'+body)
 r.update(status='paired_audio_transcript',transcript=str(v.with_suffix('.txt')),source_transcription_status=original['status'])
save()
log('BATCH FINISHED '+str({status:sum(r['status']==status for r in rows) for status in sorted(set(r['status'] for r in rows))}))
sys.exit(1 if any(r['status'] in ('failed','no_audio') for r in rows) else 0)
