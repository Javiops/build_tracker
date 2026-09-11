"""Whole-inventory constraints account for ward stacks and the role boot slot."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'scripts')]
from app.ddragon import dragon_for_patch
from qualify_patch_corpus import inventory_issues, inventory_shape, first_bot_quests

d = dragon_for_patch('16.18')
assert not inventory_issues([1036]*5 + [2055,2055],'MIDDLE',False,d)
assert 'ward_stack_overflow' in inventory_issues([2055]*3,'UTILITY',False,d)
assert not inventory_issues([1036]*6 + [2055,2055],'UTILITY',False,d)
assert 'modeled_inventory_slot_overflow' in inventory_issues([1036]*6 + [2055],'MIDDLE',False,d)
assert not inventory_issues([1036]*6 + [3006],'BOTTOM',True,d)
assert 'modeled_inventory_slot_overflow' in inventory_issues([1036]*6 + [3006],'BOTTOM',False,d)
assert 'multiple_boot_items' in inventory_issues([3006,3172],'MIDDLE',True,d)
leona = [3111,3867,3190,1029,1028,2022,2055,2055]
shape = inventory_shape(leona,'UTILITY',False,d)
assert shape['modeled_item_copies'] == 8 and shape['regular_slots'] == 6 and shape['role_slot_item_copies'] == 2
assert not inventory_issues(leona,'UTILITY',False,d)
# Five non-unique components are possible only if the regular slots fit.
assert not inventory_issues([1086]+[1036]*5,'TOP',False,d)
assert 'modeled_inventory_slot_overflow' in inventory_issues([1086]+[1036]*6,'TOP',False,d)
timeline = {'info':{'frames':[{'events':[
    {'type':'ITEM_DESTROYED','itemId':1202,'participantId':1,'timestamp':100},
    {'type':'ITEM_DESTROYED','itemId':1202,'participantId':1,'timestamp':300}]}]}}
assert first_bot_quests(timeline) == {1:100}
print('ok qualification: stack quantities, role slots and unique boots')
