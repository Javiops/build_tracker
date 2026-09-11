"""Purchase counts survive export/tensors/scoring; permutations carry no signal."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'scripts')]
import torch
import train_prefix as tp
from app.purchase_multiset import label_counts, serialize_counts, expanded_labels, TARGET_ENCODING
from app.ddragon import dragon_for_patch
from eval_policy import score_visit, serving_row

def refused(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Invalid representation was accepted')

a = {'label_ids':[1036,2055,1036,1036,1036,1036]}
b = {'label_ids':list(reversed(a['label_ids']))}
assert serialize_counts(label_counts(a)) == serialize_counts(label_counts(b)) == {'1036':5,'2055':1}
for invalid in ({'1036':0},{'1036':-1},{'1036':1.5},{'1036':True},{'01036':2}):
    refused(lambda:label_counts({'label_counts':invalid}))
refused(lambda:label_counts({'label_ids':{1036}}))
refused(lambda:label_counts({'label_counts':{'1036':5},'label_ids':[1036]}))
refused(lambda:label_counts({'target_encoding':TARGET_ENCODING,'label_id':1036}))

dragon = dragon_for_patch('16.18')
config = {'max_items':8,'max_copies':5,'inventory_encoding':'canonical-multiset-v1'}
tp.apply_config(config)
items = [3111,3867,3190,1029,1028,2022,2055,2055]
row = {'match_id':'synthetic','champion':'Tryndamere','champion_id':23,'role':'TOP','team_id':100,
       'gold':2000,'level':6,'ts':267579,'inventory':[1086], 'label_id':1036,
       'label_counts':{'1036':5},'target_encoding':TARGET_ENCODING,'save_kind':'build_spend',
       'others':[{'champion_id':89,'role':'UTILITY','team_id':100,'items':items}]}
idx = {item:i+1 for i,item in enumerate(sorted(set(items+[1036,1086])))}
def dataset(r): return tp.ShopDataset([r],{23:1,89:2},idx,{1036:0,2055:1},dragon)
left = dataset(row)
assert not label_counts(serving_row(row,2000)), 'Target counts leaked into displayed-policy featurization'
permuted = copy.deepcopy(row)
permuted['others'][0]['items'].reverse()
right = dataset(permuted)
for field in ('items','inv','query','legal','targets'):
    assert torch.equal(getattr(left,field),getattr(right,field)),field
assert left.targets[0,0].item() == 5
assert left.items[0,1].count_nonzero().item() == 8
changed = copy.deepcopy(row)
changed['label_counts']['1036'] = 4
assert dataset(changed).targets[0,0].item() == 4
changed['label_counts']['1036'] = 6
refused(lambda:dataset(changed))
too_large = copy.deepcopy(row)
too_large['others'][0]['items'].append(1036)
refused(lambda:dataset(too_large))
with tempfile.TemporaryDirectory() as temp:
    source = Path(temp)/'rows.jsonl'
    source.write_text(json.dumps(row)+'\n',encoding='utf-8')
    restored = next(tp.stream_rows(source))
    assert label_counts(restored) == Counter({1036:5})
    assert len(restored['others'][0]['items']) == 8
    assert dataset(restored).targets[0,0].item() == 5

class Policy:
    def __init__(self,ids): self.ids=ids
    def predict_options(self,*args,**kwargs):
        return [{'kind':'buy','items':[{'item_id':i,'cost':350} for i in self.ids]}]
exact,_ = score_visit(Policy([1036]*5),row,Counter({1036:5}),'build_spend',2000,dragon,None)
partial,_ = score_visit(Policy([1036]),row,Counter({1036:5}),'build_spend',2000,dragon,None)
assert exact['first_basket_exact_given_purchase'] == 1
assert partial['first_basket_exact_given_purchase'] == 0
assert partial['first_basket_recall_given_purchase'] == 0.2
# Exercise the actual displayed decoder: it must not clip five copies to the
# old four-item card limit when budget and six regular slots permit the basket.
from app.predictor import Predictor
policy = Predictor.__new__(Predictor)
policy.config = {'max_basket_items':6}
policy.dragon = dragon
policy.threshold = 0.8
policy.label_ids = [1036]
policy.label_index = {1036:0}
policy._standalone = {1036}
policy._final_components = {}
policy._probs = lambda _: (torch.tensor([0.99]),torch.tensor([0.99]),torch.tensor([5]),None)
policy._target_probs = lambda _: torch.zeros(1)
options = policy.predict_options(serving_row(row,2000),budget_slack=0)
assert Counter(item['item_id'] for item in options[0]['items']) == Counter({1036:5})
tp.apply_config({})
print('ok multiset: quantity mapping, no deduplication/clipping/truncation, permutation-invariant inputs and count-aware policy metrics')
