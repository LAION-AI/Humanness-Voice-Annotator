"""Evidence-based GENERAL/SCRIPT timing; independent of inference libraries.

Sequential events follow the Humaneness Voice timed guide. An inline burst is
a simultaneous annotation inside an inclusive segment budget, not extra elapsed
time. This explicit extension keeps every measured burst duration, including
speech-overlapping events that the historical guide rendered without numbers.
"""
from __future__ import annotations

import math
import re

FORMAT_ID = 'humaneness-general-script-v2-20261011'
PAUSE_MIN = 0.20
MAX_SEGMENT = 12.0
DURATION_STYLE = 'square'
_DURATION = re.compile(r'[\[(](\d+(?:\.\d+)?) seconds duration[\])]')
_PAUSE = re.compile(r'\[(\d+(?:\.\d+)?) seconds pause\]')
_BURST = re.compile(r'\(([^()]*?), (\d+(?:\.\d+)?) seconds\)')
_SENTENCE_END = re.compile(r'[.!?…][\"\'”’]*$')


class TimingUnavailable(ValueError):
    """The input cannot establish timed spoken segments."""


def _label(value):
    # Parentheses in directions would otherwise look like nested event tokens.
    return re.sub(r'\s+', ' ', str(value or '').replace('(', '').replace(')', '')).strip()


def _duration_tag(value, style):
    if style == 'square':
        return f'[{value:.2f} seconds duration]'
    if style == 'round':
        return f'({value:.2f} seconds duration)'
    raise ValueError('duration_style must be square or round')


def _chunks(words, events, sentences, pause_min):
    """Assign each word exactly once; sentence boundaries are never inclusive twice."""
    groups, lo = [], 0
    for i, word in enumerate(words):
        final = i == len(words) - 1
        boundary = bool(_SENTENCE_END.search(word['w']))
        if not final:
            a, b = word['end'], words[i + 1]['start']
            boundary |= b - a >= pause_min
            boundary |= any(a <= e['start_s'] and e['end_s'] <= b for e in events)
            boundary |= word['end'] - words[lo]['start'] >= MAX_SEGMENT
            # ASR sometimes omits punctuation; use a supplied sentence boundary.
            boundary |= any(a - 1e-6 <= s.get('end', -1) < b - 1e-6 for s in sentences
                            if isinstance(s.get('end'), (int, float)))
        if final or boundary:
            piece = words[lo:i + 1]
            if piece[-1]['end'] <= piece[0]['start']:
                if groups:
                    groups[-1]['words'].extend(piece)
                    groups[-1]['text'] = ' '.join(w['w'] for w in groups[-1]['words'])
                else:
                    continue
            else:
                start, end = piece[0]['start'], max(w['end'] for w in piece)
                candidates = [(max(0., min(end, s.get('end', end)) - max(start, s.get('start', start))), s)
                              for s in sentences if isinstance(s.get('start'), (int, float))
                              and isinstance(s.get('end'), (int, float))]
                cue = max(candidates, key=lambda x: x[0])[1].get('caption', '') if candidates else ''
                groups.append({'kind': 'speech', 'start_s': start, 'end_s': end,
                               'words': piece, 'text': ' '.join(w['w'] for w in piece), 'cue': _label(cue)})
            lo = i + 1
    if words and ' '.join(g['text'] for g in groups) != ' '.join(w['w'] for w in words):
        raise TimingUnavailable('A zero-duration word group cannot be timed without losing text')
    return groups


def render_timed(sentences, words, bursts, duration_s, *, pause_min=PAUSE_MIN,
                 duration_style=DURATION_STYLE, include_cues=True):
    """Return lines and an auditable plan, using measured word/event intervals.

    Every selected burst is printed once with its full observed duration. If
    intervals overlap, a containing duration budget includes the inline events;
    it is never increased by their durations. Exact endpoints stay in the plan.
    """
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError('Invalid audio duration')
    if not words and any(str(row.get('text', '')).strip() for row in sentences or []):
        raise TimingUnavailable('Spoken text is present without measured word intervals')
    timed_words = []
    for row in words or []:
        if row.get('start') is None or row.get('end') is None or not row.get('w'):
            raise TimingUnavailable('Every spoken token requires an onset and offset')
        start, end = float(row['start']), float(row['end'])
        if not math.isfinite(start + end) or start < 0 or end < start or end > duration_s + .051:
            raise TimingUnavailable('Word interval is outside the supplied recording')
        timed_words.append({**row, 'start': start, 'end': min(end, duration_s), 'w': str(row['w'])})
    if any(a['start'] > b['start'] for a, b in zip(timed_words, timed_words[1:])):
        raise TimingUnavailable('Word onsets are not in chronological order')
    events = []
    for index, row in enumerate(bursts or []):
        start = float(row.get('start_s', row.get('start', -1)))
        end = float(row.get('end_s', row.get('end', -1)))
        if not math.isfinite(start + end) or not 0 <= start < end <= duration_s + .001:
            raise ValueError('Burst requires valid start AND end seconds')
        label = _label(row.get('written') or row.get('label') or 'Vocal Burst')
        events.append({'kind': 'burst', 'start_s': start, 'end_s': min(end, duration_s),
                       'label': label, 'event_index': index, 'duration_s': min(end, duration_s) - start})
    groups = _chunks(timed_words, events, sentences or [], pause_min)
    if not groups and timed_words:
        raise TimingUnavailable('No positive-duration spoken interval')
    # Connected interval components handle overlap without double-counting time.
    components = []
    for item in sorted(groups + events, key=lambda x: (x['start_s'], x['end_s'], x['kind'])):
        if components and item['start_s'] < components[-1]['end_s'] - 1e-7:
            components[-1]['items'].append(item)
            components[-1]['end_s'] = max(components[-1]['end_s'], item['end_s'])
        else:
            components.append({'start_s': item['start_s'], 'end_s': item['end_s'], 'items': [item]})
    plan = []
    for component in components:
        speech = sorted((i for i in component['items'] if i['kind'] == 'speech'), key=lambda x: x['start_s'])
        local_events = sorted((i for i in component['items'] if i['kind'] == 'burst'), key=lambda x: x['start_s'])
        if speech:
            cuts = [component['start_s']]
            cuts.extend((a['end_s'] + b['start_s']) / 2 for a, b in zip(speech, speech[1:]))
            cuts.append(component['end_s'])
            attached = [[] for _ in speech]
            for event in local_events:
                best = max(range(len(speech)), key=lambda j:
                           max(0., min(event['end_s'], speech[j]['end_s']) - max(event['start_s'], speech[j]['start_s'])))
                attached[best].append(event)
            for j, group in enumerate(speech):
                plan.append({**group, 'start_s': cuts[j], 'end_s': cuts[j + 1], 'inline_events': attached[j],
                             'duration_includes_inline_events': True})
        elif len(local_events) == 1:
            plan.append({**local_events[0], 'inline_events': []})
        else:
            plan.append({'kind': 'burst_group', 'start_s': component['start_s'], 'end_s': component['end_s'],
                         'inline_events': local_events, 'duration_includes_inline_events': True,
                         'cue': 'overlapping non-speech vocal events', 'text': ''})
    # Small gaps are included in a neighbouring speech budget; burst lengths are
    # never changed to absorb silence or compensate for rounding.
    timeline, cursor = [], 0.
    for item in plan:
        item = dict(item)
        gap = item['start_s'] - cursor
        if gap > 1e-7:
            if gap + 1e-7 < pause_min and item['kind'] == 'speech':
                item['start_s'] = cursor
            elif gap + 1e-7 < pause_min and timeline and timeline[-1]['kind'] == 'speech':
                timeline[-1]['end_s'] = item['start_s']
            else:
                timeline.append({'kind': 'pause', 'start_s': cursor, 'end_s': item['start_s']})
        timeline.append(item)
        cursor = item['end_s']
    tail = duration_s - cursor
    if tail > 1e-7:
        if tail + 1e-7 < pause_min and timeline and timeline[-1]['kind'] == 'speech':
            timeline[-1]['end_s'] = duration_s
        else:
            timeline.append({'kind': 'pause', 'start_s': cursor, 'end_s': duration_s})
    lines = []
    for item in timeline:
        # Difference of rounded boundaries telescopes to the rounded clip length.
        budget = round(item['end_s'], 2) - round(item['start_s'], 2)
        if budget <= 0:
            if item['kind'] == 'pause':
                continue  # sub-centisecond gap has zero displayed timeline budget
            raise TimingUnavailable('A timeline interval is smaller than display precision')
        if item['kind'] == 'pause':
            lines.append(f'[{budget:.2f} seconds pause]')
        elif item['kind'] == 'burst':
            lines.append(f"({item['label']}, {budget:.2f} seconds)")
        else:
            inline = sorted(item['inline_events'], key=lambda e: (e['start_s'], e['event_index']))
            parts, i = [], 0
            for word in item.get('words', []):
                while i < len(inline) and (inline[i]['start_s'] + inline[i]['end_s']) / 2 <= word['start']:
                    e = inline[i]; parts.append(f"({e['label']}, {e['duration_s']:.2f} seconds)"); i += 1
                parts.append(word['w'])
            for e in inline[i:]:
                parts.append(f"({e['label']}, {e['duration_s']:.2f} seconds)")
            cue = f"({item['cue']}) " if include_cues and item.get('cue') else ''
            lines.append(cue + _duration_tag(budget, duration_style) + ' ' + ' '.join(parts))
    transcript = ' '.join(w['w'] for w in timed_words)
    errors = verify_timed('\n'.join(lines), transcript, duration_s, expected_bursts=len(events))
    if errors:
        raise TimingUnavailable('; '.join(errors))
    audit = {'format_id': FORMAT_ID, 'duration_style': duration_style, 'pause_min_s': pause_min,
             'timing_source': 'stored ASR word endpoints and detector event endpoints',
             'n_words': len(timed_words), 'n_bursts': len(events),
             'n_inline_overlapping_events': sum(len(x.get('inline_events', [])) for x in timeline),
             'n_speech_segments': sum(x['kind'] == 'speech' for x in timeline),
             'timeline': [{k: v for k, v in x.items() if k not in {'words', 'cue'}} for x in timeline],
             'numeric_budget_s': numeric_budget('\n'.join(lines)),
             'verified': True, 'transcript_reference': 'all ASR words, each exactly once',
             'overlap_policy': 'Inline durations are included in the enclosing segment budget, not added again'}
    return lines, audit


def numeric_budget(script):
    """Parse printed budgets only; no renderer state or original time intervals."""
    total = 0.
    for line in script.splitlines():
        durations = _DURATION.findall(line)
        total += sum(float(x) for x in durations) + sum(float(x) for x in _PAUSE.findall(line))
        if not durations:
            total += sum(float(x[1]) for x in _BURST.findall(line))
    return total


def verify_timed(script, transcript, duration_s, *, expected_bursts=None):
    errors = []
    if abs(numeric_budget(script) - float(duration_s)) > .015:
        errors.append('printed timeline does not match audio duration')
    stripped = re.sub(r'\[[^\]]*\]', ' ', script)
    stripped = re.sub(r'\([^()]*\)', ' ', stripped)
    if ' '.join(stripped.split()) != ' '.join((transcript or '').split()):
        errors.append('transcript words were lost, duplicated or reordered')
    if expected_bursts is not None and len(_BURST.findall(script)) != expected_bursts:
        errors.append('some selected bursts are missing a numeric duration')
    for line in script.splitlines():
        if line.strip() and not (_DURATION.search(line) or _PAUSE.fullmatch(line.strip()) or _BURST.fullmatch(line.strip())):
            errors.append('a script line has no duration or valid pause tag')
    return errors


def format_record(result, *, duration_style=DURATION_STYLE, include_cues=True):
    """Update the prompt surface from stored inference; no ASR/model call."""
    bursts = [dict(e, label=e.get('written') or e.get('label')) for e in result.get('variant_a_bursts', [])]
    lines, audit = render_timed(result.get('sentences', []), result.get('words', []), bursts,
                               result['dur'], duration_style=duration_style, include_cues=include_cues)
    result.update(script_timed=lines, timed_ok=True, script_timing=audit, prompt_format_id=FORMAT_ID)
    return 'GENERAL: ' + result['global_caption'] + '\nSCRIPT:\n' + '\n'.join(lines)
