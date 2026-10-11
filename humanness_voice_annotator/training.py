"""Reuse complete dataset annotations for a GENERAL/SCRIPT fine-tuning export."""
import copy
import gzip
import hashlib
import io
import json
import re
import tarfile

from .formatting import compose_annotation, sources
from .upstream import procedural_source


def records(path):
    """Sequential WebDataset MP3/JSON pairs; no extracted per-sample files."""
    pending = {}
    with tarfile.open(path, 'r|') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(('.mp3', '.json')):
                continue
            key, extension = member.name.rsplit('.', 1)
            pending.setdefault(key, {})[extension] = archive.extractfile(member).read()
            if len(pending[key]) == 2:
                pair = pending.pop(key)
                yield json.loads(pair['json']), pair['mp3']
    if pending:
        raise ValueError('Unpaired audio/annotation archive members')


class CaptionComposer:
    def __init__(self, procedural_dir=None):
        root = procedural_source(procedural_dir)
        from humaneness_caption_settings import load_settings
        self.baseline = json.loads((root/'baseline_stats_humaneness_medium_speech.json').read_text())
        self.settings = load_settings()
        self.settings.update(id='humanness-voice-annotator-v1', threshold=.6)

    def __call__(self, row, audio):
        from refresh_saved_caption import refresh_saved_caption
        digest = hashlib.sha256(audio).hexdigest()
        if digest != row['audio']['sha256']:
            raise ValueError('Original MP3 hash differs from its dataset sidecar')
        acoustic = copy.deepcopy(row['procedural'])
        refresh_saved_caption(acoustic, self.baseline, self.settings)
        acoustic.update(audio_sha256=digest, comparison_id=row['id'], language=row['language'])
        acoustic['source_metadata'] = {**row['performance'], 'source_dataset':'laion/german-english-voice-acting-challenges'}
        moss = copy.deepcopy(row['moss'])
        moss['audio_sha256'] = digest
        # Preserve the versions which actually produced the reused model outputs.
        models = sources()['models']
        original = moss['models']
        models['moss_voice'] = {'repo':original['voice_repo'], 'revision':original['voice_revision']}
        models['moss_burst'] = {'repo':original['burst_repo'], 'revision':original['burst_revision']}
        return compose_annotation(acoustic, moss, model_sources=models)


def split_group(row):
    metadata = row['generation_metadata']
    return str(row['performance']['source_subset'])+':'+str(metadata['challenge_id'])


def split_for(group, heldout_groups=()):
    """Same scenario stays in one partition across English/German and voices."""
    if group in heldout_groups:
        return 'test'
    value = int.from_bytes(hashlib.sha256(('20261011:'+group).encode()).digest()[:8], 'big') % 10000
    return 'train' if value < 9600 else 'validation' if value < 9800 else 'test'


def legacy_square(prompt):
    """An explicit adapter to the older Humaneness Voice speech-duration surface."""
    return re.sub(r'^\(([^\n]*?); (\d+(?:\.\d+)?) seconds duration\) ',
                  lambda m:'('+m[1]+') ['+m[2]+' seconds duration] ', prompt, flags=re.MULTILINE)


def training_fields(row, *, surface='general-script', legacy=False):
    if surface not in {'general-script', 'caption-transcript'}:
        raise ValueError('Unknown training surface')
    prompt = row['prompt'] if surface == 'general-script' else row['caption_transcript_prompt']
    if legacy and surface == 'general-script':
        prompt = legacy_square(prompt)
    text = prompt.split('SCRIPT:\n', 1)[1].strip() if surface == 'general-script' else row['asr_transcript']
    return {'instruction':prompt, 'text':text, 'tokens':row['codec']['frames'],
            'language':{'en':'English','de':'German'}[row['language']]}


def tuning_records(path, *, split=None, include_annotation=False):
    """Read a token shard, validating each payload; yields (metadata, codes).

    Codes are integer [frames, 12] arrays. Optional canonical annotations are
    attached to metadata['annotation']; original MP3s remain in the audio shards.
    """
    import numpy as np
    suffixes = ('.annotation.json.gz', '.codes.npy', '.json')
    pending = {}
    with tarfile.open(path, 'r|') as archive:
        for member in archive:
            suffix = next((s for s in suffixes if member.name.endswith(s)), None)
            if not member.isfile() or suffix is None:
                continue
            key = member.name[:-len(suffix)]
            pair = pending.setdefault(key, {})
            if suffix in pair:
                raise ValueError('Duplicate training archive member')
            pair[suffix] = archive.extractfile(member).read()
            if len(pair) != 3:
                continue
            pending.pop(key)
            row = json.loads(pair['.json'])
            if row['id'] != key or row['schema'] != 'humanness-voice-tuning-v1':
                raise ValueError('Unexpected training key/schema')
            for name, digest in [('.codes.npy', row['codec']['codes_file_sha256']),
                                 ('.annotation.json.gz', row['annotation_file_sha256'])]:
                if hashlib.sha256(pair[name]).hexdigest() != digest:
                    raise ValueError('Training payload hash mismatch')
            codes = np.load(io.BytesIO(pair['.codes.npy']), allow_pickle=False)
            frames = row['codec']['frames']
            if (codes.shape != (frames, 12) or codes.dtype != np.dtype('<i2')
                    or frames < 1 or codes.min() < 0 or codes.max() > 1023):
                raise ValueError('Invalid codec target array')
            if frames*3840 != (row['codec']['original_samples_48khz']
                               + row['codec']['right_padding_samples']):
                raise ValueError('Invalid codec tail padding')
            if split and row['split'] != split:
                continue
            if include_annotation:
                annotation = json.loads(gzip.decompress(pair['.annotation.json.gz']))
                if annotation['annotation_sha256'] != row['annotation_sha256']:
                    raise ValueError('Mismatched canonical annotation')
                row['annotation'] = annotation
            yield row, codes
    if pending:
        raise ValueError('Incomplete training archive record')


def pack_target(processor, model_config, row, codes, *, reference_codes=None,
                surface='general-script', legacy=False):
    """Native shifted teacher-forcing targets for Humaneness Voice Small.

    Requires a processor accepting precomputed [T,12] integer codes. Loads no
    neural codec. User instructions and reference audio never receive a loss.
    """
    import numpy as np
    import torch
    codes = torch.as_tensor(np.array(codes, dtype=np.int64, copy=True))
    if codes.shape != (row['codec']['frames'], int(model_config.n_vq)):
        raise ValueError('Target shape disagrees with the model or frame budget')
    fields = training_fields(row, surface=surface, legacy=legacy)
    ref = None if reference_codes is None else [torch.as_tensor(
        np.array(reference_codes, dtype=np.int64, copy=True))]
    user = processor.build_user_message(**fields, reference=ref)
    assistant = processor.build_assistant_message(audio_codes_list=[codes])
    ids = processor([[user, assistant]], mode='computing_loss')['input_ids'][0].long()
    labels = torch.full_like(ids, -100)
    labels[:-1] = ids[1:]
    target_text = labels[:, 0]
    slot = int(model_config.audio_assistant_slot_token_id)
    end = int(model_config.audio_end_token_id)
    supervised = (target_text == slot) | (target_text == end)
    first = (ids[:, 0] == slot).nonzero()
    if not first.numel():
        raise ValueError('Processor did not emit assistant audio')
    supervised[:max(0, int(first[0, 0])-1)] = False
    labels[~supervised] = -100
    audio = labels[:, 1:]
    audio[(audio >= int(model_config.audio_pad_token_id)) | (audio < 0)] = -100
    if int(supervised.sum()) != len(codes)+1:
        raise ValueError('Expected one continuation per frame plus one audio stop')
    return {'input_ids':ids, 'labels':labels, 'n_supervised_steps':int(supervised.sum())}
