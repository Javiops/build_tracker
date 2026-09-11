"""Read frozen raw events; quantify ambiguous undo and support observations."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.dataset_artifacts import sha256, write_json
from app.riot import RawMatchArchive

corpus = ROOT / 'data/corpora/16.18_initial'
archive = RawMatchArchive(corpus / 'raw')
counts, affected, examples = Counter(), set(), []
for line in (corpus / 'source_inventory.jsonl').read_text().splitlines():
    mid = json.loads(line)['match_id']
    timeline = archive.read('timeline', mid)
    counts['games'] += 1
    for frame in timeline['info']['frames']:
        for event in frame.get('events', []):
            kind = event.get('type')
            if kind == 'ITEM_UNDO':
                counts['undos'] += 1
                if 'goldGain' not in event:
                    counts['undo_missing_gold_gain'] += 1
                if not event.get('beforeId') and not event.get('afterId'):
                    counts['undo_without_item_ids'] += 1
                    affected.add(mid)
                    if len(examples) < 10:
                        examples.append({'match_id':mid, 'event':event})
            if kind == 'ITEM_PURCHASED' and event.get('itemId') in (3869,3870,3871,3876,3877):
                counts['observed_support_final_purchase'] += 1
out = ROOT / 'data/acceptance/16.18_initial/source_ambiguities.json'
write_json(out, {'corpus_manifest_sha256':sha256(corpus / 'corpus_manifest.json'),
    'counts':dict(counts), 'games_with_opaque_undos':sorted(affected), 'examples':examples})
print(json.dumps({'report':str(out),'counts':dict(counts),'games_with_opaque_undos':len(affected),'examples':examples[:3]},indent=2))
