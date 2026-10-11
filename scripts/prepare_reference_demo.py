"""Rebuild the 50-clip annotator demo from the frozen paired-comparison artifacts."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from humanness_voice_annotator import compose_annotation
from humanness_voice_annotator.upstream import procedural_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--procedural-dir',type=Path)
    args=p.parse_args()
    source=procedural_source(args.procedural_dir)
    from refresh_saved_caption import refresh_saved_caption
    from humaneness_caption_settings import load_settings,filter_events
    baseline=json.loads((source/'baseline_stats_humaneness_medium_speech.json').read_text())
    settings=load_settings();settings.update(id='humanness-voice-annotator-v1',threshold=.6)
    refs={r['comparison_id']:r for r in map(json.loads,(args.source_root/'reference_samples.jsonl').open())}
    originals={r['id']:r for r in map(json.loads,(args.source_root/'samples.jsonl').open())}
    predictions={(r['sample_id'],r['task'],r['event_index']):r for path in args.source_root.glob('predictions-r*.jsonl') for r in map(json.loads,path.open())}
    rows=sorted([json.loads(x) for x in (args.source_root/'comparison_samples.jsonl').open()],key=lambda x:(x['language']!='en',x['id']))
    args.output.mkdir(parents=True,exist_ok=True)
    artifacts=args.output/'artifacts';artifacts.mkdir(exist_ok=True)
    manifest=[];annotations=[]
    for row in rows:
        sid=row['id'];original=originals[sid]
        acoustic=refresh_saved_caption(copy.deepcopy(refs[sid]),baseline,settings)
        digest=hashlib.sha256(Path(row['audio_path']).read_bytes()).hexdigest()
        assert digest==acoustic['audio_sha256']==original['sha256']
        selected=filter_events(acoustic['humaneness_ears_medium']['burst_event_proposals'],settings)
        assert [(e['start_s'],e['end_s']) for e in selected]==[(e['start_s'],e['end_s']) for e in original['events']]
        moss={'audio_sha256':digest,'audio_duration_s':row['duration_s'],
              'voice_caption':predictions[(sid,'caption',None)],
              'burst_captions':[dict(e,caption=predictions[(sid,'burst_caption',e['event_index'])]) for e in original['events']],
              'locator_settings':{'threshold':.6,'min_duration_s':.08,'nms_overlap_fraction':.5,'event_cap':None}}
        annotation=compose_annotation(acoustic,moss)
        annotations.append(annotation)
        (artifacts/(sid+'.json')).write_text(json.dumps(annotation,ensure_ascii=False,indent=2)+'\n')
        manifest.append({'id':sid,'language':row['language'],'duration_s':row['duration_s'],
                         'audio_path':row['audio_path'],'audio_sha256':digest,
                         'annotation_path':str(artifacts/(sid+'.json'))})
        if len(annotations)==1:
            examples=Path(__file__).resolve().parents[1]/'examples';examples.mkdir(exist_ok=True)
            saved=copy.deepcopy(acoustic)
            for key in ['audio_path','source_metadata']:saved.pop(key,None)
            saved['source_metadata']=annotation['audio']['source_metadata']
            (examples/'acoustic.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n')
            (examples/'moss.json').write_text(json.dumps(moss,ensure_ascii=False,indent=2)+'\n')
            (examples/'annotation.json').write_text(json.dumps(annotation,ensure_ascii=False,indent=2)+'\n')
            (examples/'annotation.txt').write_text(annotation['prompt']+'\n')
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in manifest))
    (args.output/'annotations.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in annotations))
    public=Path(__file__).resolve().parents[1]/'docs/data';public.mkdir(parents=True,exist_ok=True)
    with gzip.open(public/'annotations.jsonl.gz','wt',encoding='utf-8') as f:
        for annotation in annotations:f.write(json.dumps(annotation,ensure_ascii=False)+'\n')
    summary={'n_recordings':len(rows),'languages':{'en':sum(x['language']=='en' for x in rows),'de':sum(x['language']=='de' for x in rows)},
             'source_sentences':sum(len(x['raw_model_outputs']['sentence_predictions']) for x in annotations),
             'timed_speech_segments':sum(len(x['segments']) for x in annotations),
             'selected_burst_intervals':sum(len(x['burst_events']) for x in annotations),
             'words_preserved':sum(len(x['words']) for x in annotations),
             'all_50_verified':all(x['timing']['independent_verification']['verified'] for x in annotations),
             'new_neural_inference_for_this_demo':False,'saved_predictions_reused':True,
             'format':'GENERAL / SCRIPT','format_version':'humanness-general-script-v1'}
    (public/'format-verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
