"""Check token loading and native TTS labels without loading neural weights."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from humanness_voice_annotator.training import tuning_records, pack_target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path,help='Downloaded Humaneness Voice Small repository')
    p.add_argument('--shard',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    from transformers import AutoTokenizer
    spec=importlib.util.spec_from_file_location('humaneness_small_packing',args.bundle/'code/moss_small.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SFT3=str(args.bundle/'assets/sft3')
    module.QWEN=str(args.bundle/'assets/qwen3')
    _,_,Processor=module.export_classes()
    config=module.make_config('M1')  # Same native text/codebook contract as the released Small packer.
    tokenizer=AutoTokenizer.from_pretrained(module.SFT3,local_files_only=True)
    processor=Processor(tokenizer=tokenizer,audio_tokenizer=None,model_config=config)
    iterator=tuning_records(args.shard,include_annotation=True)
    examples=[next(iterator) for _ in range(2)]
    results=[]
    for i,(row,codes) in enumerate(examples):
        reference=examples[1-i][1][:24]  # Different recording, for masking mechanics only.
        for surface,legacy,ref in [('general-script',False,None),('general-script',False,reference),
                                    ('general-script',True,None),('caption-transcript',False,None)]:
            packed=pack_target(processor,config,row,codes,reference_codes=ref,surface=surface,legacy=legacy)
            audio_targets=int((packed['labels'][:,1:]>=0).sum())
            if audio_targets!=12*len(codes):
                raise ValueError('Masked audio target count changed')
            results.append({'id':row['id'],'surface':surface,'legacy':legacy,'reference_frames':0 if ref is None else len(ref),
                            'codec_frames':len(codes),'input_shape':list(packed['input_ids'].shape),
                            'supervised_steps':packed['n_supervised_steps'],'audio_loss_targets':audio_targets})
    report={'schema':'humanness-voice-native-packing-check-v1','passed':True,
            'processor_has_chat_template':bool(tokenizer.chat_template),
            'chat_template_policy':'Use released processor behavior, including its raw-content fallback if absent',
            'neural_codec_loaded':False,'neural_tts_weights_loaded':False,'results':results}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
