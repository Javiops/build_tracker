"""Bounded inspection of raw event types and item-related payload shapes."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.riot import RawMatchArchive
from app.dataset_artifacts import write_json

corpus = ROOT / 'data/corpora/16.18_initial'
archive = RawMatchArchive(corpus / 'raw')
counts, examples = Counter(), {}
for line in (corpus / 'source_inventory.jsonl').read_text().splitlines():
    mid = json.loads(line)['match_id']
    for f in archive.read('timeline',mid)['info']['frames']:
        for event in f.get('events',[]):
            kind = event.get('type','')
            counts[kind] += 1
            if kind not in examples and ('ITEM' in kind or 'WARD' in kind or 'QUEST' in kind):
                examples[kind] = {'match_id':mid,'event':event}
report = {'counts':dict(counts),'examples':examples}
write_json(ROOT / 'data/acceptance/16.18_initial/event_schema.json',report)
print(json.dumps(report,indent=2))
