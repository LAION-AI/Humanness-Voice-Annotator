"""Encode MP3/JSON dataset shards and add verified Humanness Voice training data.

One persistent codec per GPU worker, streaming batches, atomic per-source-shard
outputs. Existing audio and annotations are never rewritten. No caption-model
inference is required when the input has complete MOSS/acoustic sidecars.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanness_voice_annotator.training import CaptionComposer, records, split_group, split_for

ROOT = Path(__file__).resolve().parents[1]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(encoded(value)+b'\n')
    temporary.replace(path)


def add_member(tar, name, payload):
    member = tarfile.TarInfo(name)
    member.size = len(payload); member.mode = 0o644; member.mtime = 0
    tar.addfile(member, io.BytesIO(payload))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Existing dataset root with inventory.json and data/')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--codec-dir', required=True)
    p.add_argument('--procedural-dir')
    p.add_argument('--source-revision', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--chunk-duration', type=float, default=4.8)
    p.add_argument('--rank', type=int, default=int(os.environ.get('SLURM_PROCID', 0)))
    p.add_argument('--world-size', type=int, default=int(os.environ.get('SLURM_NTASKS', 1)))
    p.add_argument('--limit-samples-per-shard', type=int, default=0, help='Smoke test only; use a separate output root')
    p.add_argument('--limit-shards-per-rank', type=int, default=0)
    p.add_argument('--heldout-groups', type=Path, help='JSON list of scenario IDs reserved for test')
    p.add_argument('--decode-canary', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if not 0 <= args.rank < args.world_size or args.batch_size < 1:
        p.error('Invalid rank/world size/batch size')
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from transformers import AutoModel
    torch.set_num_threads(args.threads)
    count = torch.cuda.device_count()
    local = int(os.environ.get('SLURM_LOCALID', 0))
    index = 0 if count == 1 else local if count == 4 and local < 4 else None
    if index is None:
        raise RuntimeError('Expected one visible GPU, or four GPUs with a local rank')
    torch.cuda.set_device(index)
    device = f'cuda:{index}'
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.output/'data'; data.mkdir(exist_ok=True)
    state = args.output/'state'; state.mkdir(exist_ok=True)
    codec_lock = json.loads((ROOT/'humanness_voice_annotator/assets/training-codec.json').read_text())
    heldout = set(json.loads(args.heldout_groups.read_text()) if args.heldout_groups else [])
    code_names = ['humanness_voice_annotator/training.py', 'humanness_voice_annotator/formatting.py',
                  'scripts/export_training_dataset.py']
    lock = {'schema':'humanness-voice-tuning-export-v1','source_dataset':'laion/german-english-voice-acting-challenges',
            'source_revision':args.source_revision, 'codec':codec_lock,
            'n_vq':12,'sampling_rate':48000,'downsample_rate':3840,'frame_rate':12.5,
            'codec_weight_dtype':'fp32','codec_compute_dtype':'bf16','attention':'sdpa',
            'tail_policy':'right-zero-pad-to-3840-samples-before-encoding',
            'chunk_duration_s':args.chunk_duration,'batch_size':args.batch_size,
            'input_inventory_sha256':hashlib.sha256((args.input/'inventory.json').read_bytes()).hexdigest(),
            'heldout_groups':sorted(heldout),'split_seed':20261011,
            'code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in code_names},
            'sample_limit':args.limit_samples_per_shard,'shard_limit_per_rank':args.limit_shards_per_rank}
    lockhash = hashlib.sha256(encoded(lock)).hexdigest()
    # All ranks use the same bytes; an exclusive file creation resolves startup races.
    lockpath = args.output/'export-lock.json'
    try:
        with lockpath.open('xb') as file: file.write(encoded(lock)+b'\n')
    except FileExistsError:
        for _ in range(100):
            try: stored=json.loads(lockpath.read_text()); break
            except json.JSONDecodeError: time.sleep(.05)
        if stored != lock:
            raise ValueError('Changed export contract; use a new output directory')
    inventory = json.loads((args.input/'inventory.json').read_text())
    assigned = list(enumerate(inventory))[args.rank::args.world_size]
    if args.limit_shards_per_rank:
        assigned = assigned[:args.limit_shards_per_rank]
    todo = []
    for number, source in assigned:
        proof = state/f'shard-{number:05d}.json'
        if proof.exists():
            old = json.loads(proof.read_text())
            if old['export_lock_sha256'] != lockhash or not (args.output/old['path']).exists():
                raise ValueError('Completed shard does not match this export')
            continue
        todo.append((number, source))
    if not todo:
        print('ALL_ASSIGNED_SHARDS_ALREADY_COMPLETE', args.rank, flush=True); return
    start = time.monotonic()
    # Check local reviewed source and weights once across the workers.
    for i, source in enumerate(codec_lock['files']):
        if i % args.world_size != args.rank:
            continue
        path = Path(args.codec_dir)/source['path']
        if source['lfs_sha256']:
            with path.open('rb') as file: digest=hashlib.file_digest(file,'sha256').hexdigest()
            if digest != source['lfs_sha256']: raise ValueError('Codec weight mismatch: '+source['path'])
        else:
            payload=path.read_bytes()
            digest=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
            if digest != source['git_blob']: raise ValueError('Codec source mismatch: '+source['path'])
    # Every worker verifies both code modules, even when no weight shard is assigned to it.
    for source in codec_lock['files']:
        if source['path'].endswith('.py'):
            payload=(Path(args.codec_dir)/source['path']).read_bytes()
            if hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()!=source['git_blob']:
                raise ValueError('Codec source mismatch')
    codec = AutoModel.from_pretrained(args.codec_dir, trust_remote_code=True, local_files_only=True,
                                     codec_weight_dtype='fp32').eval().to(device)
    codec.set_compute_dtype('bf16'); codec.set_attention_implementation('sdpa')
    if codec.sampling_rate != 48000 or codec.downsample_rate != 3840:
        raise ValueError('Unexpected codec rate/frame duration')
    composer = CaptionComposer(args.procedural_dir)
    print('CODEC_READY',args.rank,device,flush=True)

    def prepare(pair):
        original, audio = pair
        annotation = composer(original, audio)
        wave, sr = sf.read(io.BytesIO(audio), dtype='float32', always_2d=True)
        if wave.shape[1] != 1 or not np.isfinite(wave).all():
            raise ValueError('Expected finite mono source audio')
        mono = wave[:,0]
        if sr != 48000:
            divisor = math.gcd(sr,48000)
            mono = resample_poly(mono,48000//divisor,sr//divisor).astype(np.float32)
        if abs(len(mono)/48000-original['audio']['duration_s']) > .002:
            raise ValueError('Decoded duration disagrees with source annotations')
        original_samples = len(mono)
        # The pinned backend pads tensor storage but floors its length masks.
        # Make the partial final frame valid by padding each waveform explicitly.
        mono = np.pad(mono, (0, (-original_samples) % 3840))
        wave = torch.from_numpy(np.repeat(mono[None,:],2,axis=0))
        return original, annotation, wave, original_samples

    total = 0; total_seconds = 0.; canary_done = False
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for number, source in todo:
            input_path = args.input/source['path']
            with input_path.open('rb') as file:
                if hashlib.file_digest(file,'sha256').hexdigest()!=source['sha256']:
                    raise ValueError('Input archive hash mismatch')
            target = data/f'voice-acting-tuning-{number:05d}.tar'
            temporary = target.with_suffix('.tar.tmp')
            iterator = records(input_path)
            if args.limit_samples_per_shard:
                iterator = itertools.islice(iterator,args.limit_samples_per_shard)
            index_rows = []; counts = {'frames':0,'events':0,'words':0,'seconds':0.}
            encode_seconds = 0.
            with tarfile.open(temporary,'w') as tar:
                while True:
                    batch = list(itertools.islice(iterator,args.batch_size))
                    if not batch: break
                    prepared = list(pool.map(prepare,batch))
                    waves = [x[2].to(device) for x in prepared]
                    tick = time.monotonic()
                    with torch.inference_mode():
                        result = codec.batch_encode(waves,num_quantizers=12,chunk_duration=args.chunk_duration)
                    torch.cuda.synchronize(); encode_seconds += time.monotonic()-tick
                    if result.audio_codes.shape[0] != 12:
                        raise ValueError('Codec returned wrong codebook count')
                    for i,(original,annotation,wave,original_samples) in enumerate(prepared):
                        frames = int(result.audio_codes_lengths[i])
                        codes = result.audio_codes[:,i,:frames].T.contiguous().cpu().numpy()
                        if codes.shape != (math.ceil(wave.shape[-1]/3840),12) or codes.min()<0 or codes.max()>=1024:
                            raise ValueError(f"Invalid codec output: id={original['id']}, shape={codes.shape}, expected={(wave.shape[-1]//3840,12)}, range={(int(codes.min()),int(codes.max()))}")
                        codes = codes.astype('<i2')
                        if args.decode_canary and not canary_done:
                            with torch.inference_mode():
                                decoded=codec.batch_decode([torch.from_numpy(codes.astype(np.int64).T.copy()).to(device)],num_quantizers=12,chunk_duration=args.chunk_duration)
                            assert int(decoded.audio_lengths[0])==frames*3840 and torch.isfinite(decoded.audio).all()
                            sf.write(state/f'codec-reconstruction-r{args.rank:02d}.wav',decoded.audio[0].float().cpu().numpy().T,48000,subtype='PCM_16')
                            atomic(state/f'codec-canary-r{args.rank:02d}.json',{'id':original['id'],'frames':frames,'codebooks':12,'range':[int(codes.min()),int(codes.max())],'decode_samples':frames*3840,'finite':True})
                            canary_done=True
                        blob=io.BytesIO();np.save(blob,codes,allow_pickle=False);blob=blob.getvalue()
                        group=split_group(original); split=split_for(group,heldout)
                        row={'schema':'humanness-voice-tuning-v1','id':original['id'],'language':original['language'],
                             'voice':original['generation_metadata']['voice'],'source_subset':original['performance']['source_subset'],
                             'challenge_group':group,'split':split,'prompt':annotation['prompt'],
                             'asr_transcript':annotation['asr_transcript'],
                             'caption_transcript_prompt':original['humaneness_voice_training_formats']['variants']['whole_audio_only__plain']['prompt'],
                             'audio':original['audio'],'generation_metadata':original['generation_metadata'],
                             'codec':{'repository':codec_lock['repository'],'revision':codec_lock['revision'],'frames':frames,
                                      'n_vq':12,'shape':[frames,12],'dtype':'int16-little-endian','code_range':[0,1023],
                                      'sampling_rate':48000,'frame_duration_s':.08,'downsample_rate':3840,
                                      'mono_duplicated_to_stereo':True,'original_samples_48khz':original_samples,
                                      'right_padding_samples':frames*3840-original_samples,
                                      'codes_file_sha256':hashlib.sha256(blob).hexdigest()},
                             'format_verified':annotation['timing']['independent_verification']['verified'],
                             'asr_empty':not bool(annotation['asr_transcript'].strip()),
                             'annotation_sha256':annotation['annotation_sha256'],
                             'source_dataset':{'repository':lock['source_dataset'],'revision':args.source_revision,'shard':source['path'],'key':original['id']},
                             'export_lock_sha256':lockhash,'license':'cc-by-4.0'}
                        capsule=gzip.compress(encoded(annotation),compresslevel=6,mtime=0)
                        row['annotation_file_sha256']=hashlib.sha256(capsule).hexdigest()
                        for extension,payload in [('json',encoded(row)),('codes.npy',blob),('annotation.json.gz',capsule)]:
                            add_member(tar,row['id']+'.'+extension,payload)
                        index_rows.append({k:row[k] for k in ['id','language','voice','source_subset','challenge_group','split','asr_empty','annotation_sha256']})
                        index_rows[-1].update(frames=frames,duration_s=original['audio']['duration_s'],audio_sha256=original['audio']['sha256'],shard=str(target.relative_to(args.output)))
                        counts['frames']+=frames;counts['events']+=len(annotation['burst_events']);counts['words']+=len(annotation['words']);counts['seconds']+=original['audio']['duration_s']
                        total+=1;total_seconds+=original['audio']['duration_s']
                    atomic(state/f'progress-r{args.rank:02d}.json',{'rank':args.rank,'n_this_run':total,'audio_hours':total_seconds/3600,'elapsed_s':time.monotonic()-start,'clips_per_s':total/max(1,time.monotonic()-start),'current_shard':number,'encode_s_current_shard':encode_seconds})
            expected=min(source['n_samples'],args.limit_samples_per_shard) if args.limit_samples_per_shard else source['n_samples']
            if len(index_rows)!=expected:raise ValueError('Missing input samples')
            temporary.replace(target)
            with target.open('rb') as file: digest=hashlib.file_digest(file,'sha256').hexdigest()
            proof={'n':len(index_rows),'path':str(target.relative_to(args.output)),'sha256':digest,
                   'size_bytes':target.stat().st_size,'source':source,'export_lock_sha256':lockhash,
                   'totals':counts,'index':index_rows,'encode_seconds':encode_seconds}
            atomic(state/f'shard-{number:05d}.json',proof)
            print('TUNING_SHARD_COMPLETE',args.rank,number,len(index_rows),round(encode_seconds,2),flush=True)
    atomic(state/f'DONE-r{args.rank:02d}.json',{'rank':args.rank,'assigned_shards':len(assigned),'n_this_run':total,'elapsed_s':time.monotonic()-start,'all_assigned_complete':True})


if __name__ == '__main__':
    main()
