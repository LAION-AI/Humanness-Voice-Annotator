"""Aggregate the frozen audio-review scores with equal weight per recording."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics


def quantile(values,q):
    values=sorted(values);position=(len(values)-1)*q
    lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def summarize(rows):
    rubric=json.loads((Path(__file__).parent/'rubric.json').read_text())
    rng=random.Random(20261011)
    samples=[[rng.randrange(len(rows)) for _ in rows] for _ in range(10000)]
    results={}
    values={'composite':[r['weighted_score'] for r in rows]}
    values.update({k:[r['answer']['ratings'][k] for r in rows] for k in rubric['definitions']})
    for key,xs in values.items():
        means=[statistics.mean(xs[i] for i in ids) for ids in samples]
        results[key]={'n':len(xs),'mean':statistics.mean(xs),'bootstrap_95_ci':[quantile(means,.025),quantile(means,.975)],
                      'recordings_rated_0_to_2':sum(x<=2 for x in xs)}
    local=[x for r in rows for x in r['answer']['sentence_reviews']]
    return {'rubric_id':rubric['id'],'n_recordings':len(rows),'languages':dict(Counter(r['language'] for r in rows)),
            'model_versions':sorted({r['model_version'] for r in rows}),'results':results,
            'local_segment_reviews':len(local),'local_segment_diagnostics':{
                'emotion_fit_mean':statistics.mean(x['emotion_fit'] for x in local),
                'delivery_fit_mean':statistics.mean(x['delivery_fit'] for x in local),
                'aggregation':'Unweighted across local segments; diagnostic only, excluded from the clip-weighted composite'},
            'estimated_list_price_usd':sum(r['estimated_list_price_usd'] for r in rows),
            'usage_totals':{k:sum(r['usage'].get(k,0) for r in rows) for k in ['promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount']},
            'pricing':{'date':'2026-10-11','input_usd_per_million':.75,'output_usd_per_million':3.75,
                       'source':'https://ai.google.dev/gemini-api/docs/pricing','not_an_invoice':True},
            'rubric':rubric,'automated_reviewer_not_human_ground_truth':True,
            'bootstrap':'10,000 clip resamples, seed 20261011; excludes reviewer bias and domain shift',
            'retuning_using_these_scores':False,'new_rubric_not_comparable_to_previous_composite':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviews',type=Path,required=True);p.add_argument('--output','--out',type=Path,required=True)
    a=p.parse_args()
    if a.reviews.is_dir():
        rows=[json.loads(path.read_text()) for path in sorted(a.reviews.glob('*.json'))]
    else:
        rows=list(map(json.loads,a.reviews.read_text().splitlines()))
    if not rows:
        p.error('No completed review records found')
    result=summarize(rows);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['mean'] for k,v in result['results'].items()},indent=2))
    print('Estimated review cost USD',result['estimated_list_price_usd'])
