"""Causal item-event rules, including timestamped role-quest transitions.

Inventories represent owned build items, including a boot in the role slot.
No final match inventory is read. Support choices absent from the timeline are
represented by a marked Bounty tier proxy in reconstruct._items_payload.
"""
from collections import Counter

SUPPORT_ATLAS, SUPPORT_COMPASS, SUPPORT_BOUNTY = 3865, 3866, 3867
SUPPORT_FINISHED = {3869, 3870, 3871, 3876, 3877}
SUPPORT_LINE = {SUPPORT_ATLAS, SUPPORT_COMPASS, SUPPORT_BOUNTY} | SUPPORT_FINISHED


class InventoryReplay:
    def __init__(self, inventories, dragon):
        self.inventories = inventories
        self.dragon = dragon
        self.boot_upgrades = dragon.free_boot_upgrades()
        self.mid_complete = set()
        self.bot_complete = set()
        self.pending_boots = {}
        self.transformations = {}
        for raw_id, item in dragon._items['data'].items():
            source = item.get('specialRecipe')
            if source and item.get('maps', {}).get('11') and item.get('gold', {}).get('purchasable') is False:
                source = int(source)
                if source in self.transformations:
                    raise ValueError(f'Ambiguous automatic item transformation for {source}')
                self.transformations[source] = int(raw_id)

    def effective_purchase(self, owner, item):
        return self.boot_upgrades.get(item, item) if owner in self.mid_complete else item

    def _is_boot(self, item):
        cls = self.dragon.classify(item)
        return cls['is_boots'] or cls['is_basic_boots']

    def _consumes(self, purchase, component):
        if purchase == component:
            return False
        pending, seen = list(self.dragon.from_ids(purchase)), set()
        while pending:
            item = pending.pop()
            if item == component:
                return True
            if item not in seen:
                seen.add(item)
                pending.extend(self.dragon.from_ids(item))
        return self.transformations.get(component) == purchase

    @staticmethod
    def _remove(inventory, item):
        if item in inventory:
            inventory.remove(item)

    def apply_batch(self, owner, ts, batch, undo_restores=None):
        inventory = self.inventories[owner]
        undo_restores = undo_restores or {}
        purchases = {e.get('itemId') for e in batch if e.get('type') == 'ITEM_PURCHASED'}
        destroyed = {e.get('itemId') for e in batch if e.get('type') == 'ITEM_DESTROYED'}
        mid_now = 1201 in destroyed
        bot_now = 1202 in destroyed
        if mid_now:
            self.mid_complete.add(owner)
        if bot_now:
            self.bot_complete.add(owner)
        if mid_now or bot_now:
            # A quest and its slot-transfer destruction can be separated by
            # another player's event at the same timestamp.
            inventory.extend(self.pending_boots.pop((owner, ts), []))
        if owner in self.mid_complete:
            inventory[:] = [self.boot_upgrades.get(i, i) for i in inventory]

        bought = Counter()
        for event in batch:
            kind, item = event.get('type'), event.get('itemId') or 0
            if kind == 'ITEM_PURCHASED':
                effective = self.effective_purchase(owner, item)
                if effective in SUPPORT_LINE and effective in inventory:
                    continue
                # A source may report the free upgrade explicitly after the
                # quest has already transformed the held boots.
                if self._is_boot(effective) and effective in inventory:
                    continue
                if effective in SUPPORT_FINISHED:
                    inventory[:] = [i for i in inventory if i not in SUPPORT_LINE]
                inventory.append(effective)
                if not self.dragon.classify(effective)['target_skip']:
                    bought[effective] += 1
            elif kind in ('ITEM_DESTROYED', 'ITEM_SOLD'):
                if kind == 'ITEM_DESTROYED' and item in (1201, 1202):
                    continue
                consumed = any(self._consumes(p, item) for p in purchases if p)
                if kind == 'ITEM_DESTROYED' and self._is_boot(item) and not consumed:
                    if owner in self.bot_complete or (owner in self.mid_complete and item in self.boot_upgrades):
                        # Moving to the role slot / automatic upgrade does not
                        # discard the owned build item.
                        continue
                    if item in inventory:
                        self.pending_boots.setdefault((owner, ts), []).append(item)
                if kind == 'ITEM_DESTROYED' and item in SUPPORT_LINE:
                    self._remove(inventory, item)
                    upgrade = {3865:3866, 3866:3867, 3867:3867}.get(item)
                    if upgrade and upgrade not in inventory:
                        inventory.append(upgrade)
                    if item in SUPPORT_FINISHED:
                        self._remove(inventory, SUPPORT_BOUNTY)
                    continue
                if kind == 'ITEM_SOLD' and item not in inventory:
                    item = self.effective_purchase(owner, item)
                    if item in SUPPORT_FINISHED and SUPPORT_BOUNTY in inventory:
                        item = SUPPORT_BOUNTY
                held = item in inventory
                self._remove(inventory, item)
                if kind == 'ITEM_DESTROYED' and held and not consumed and item in self.transformations:
                    inventory.append(self.transformations[item])
            elif kind == 'ITEM_UNDO':
                before, after = event.get('beforeId') or 0, event.get('afterId') or 0
                if before:
                    self._remove(inventory, before if before in inventory else self.effective_purchase(owner, before))
                restored = list(undo_restores.get(id(event), []))
                if after and after not in restored:
                    restored.append(after)
                inventory.extend(self.effective_purchase(owner, i) for i in restored)
        return bought

    def consume_control_ward(self, owner):
        inventory = self.inventories.get(owner, [])
        for item in (2055, 772043):
            if item in inventory:
                inventory.remove(item)
                return
