"""Shop targets are item -> positive copy count mappings; order is irrelevant."""
from collections import Counter
from collections.abc import Mapping

TARGET_ENCODING = 'item-counts-v1'


def label_counts(row):
    """Read explicit multisets, with lossless support for forensic legacy rows."""
    if 'label_counts' in row and row['label_counts'] is not None:
        raw = row['label_counts']
        if not isinstance(raw, Mapping):
            raise ValueError('label_counts must be an item-to-count mapping')
        counts = Counter()
        for key, count in raw.items():
            if isinstance(key,bool) or not str(key).isdigit() or int(key) <= 0:
                raise ValueError('Multiset item IDs must be positive integers')
            item = int(key)
            if str(item) != str(key) or item in counts or type(count) is not int or count <= 0:
                raise ValueError('Multiset keys must be canonical and counts positive integers')
            counts[item] = count
        if row.get('label_ids') is not None and Counter(row['label_ids']) != counts:
            raise ValueError('Legacy labels disagree with the explicit multiset')
        if row.get('target_encoding') not in (None,TARGET_ENCODING):
            raise ValueError('Unknown target encoding')
        return counts
    if row.get('target_encoding') == TARGET_ENCODING:
        raise ValueError('Missing explicit purchase multiset')
    ids = row.get('label_ids')
    if ids is None:
        ids = [row['label_id']] if row.get('label_id') else []
    if isinstance(ids,(set,Mapping)):
        raise ValueError('A set cannot represent purchase multiplicity')
    return Counter(int(item) for item in ids if item)


def serialize_counts(counts):
    result = {str(item):count for item,count in sorted(counts.items())}
    label_counts({'label_counts':result})
    return result


def expanded_labels(row):
    """Deterministic tensor adapter; repetition preserves every multiset count."""
    return [item for item,count in sorted(label_counts(row).items()) for _ in range(count)]
