"""Removes the empty space between packed boxes.

The layer-based GA leaves small gaps: a layer is as deep as its deepest box, so shallower
boxes in it stop short of the next layer, and boxes stacked on short neighbours can end up
floating. This post-processing step slides every box as far as it can go, first down
(gravity), then towards the cab (x = 0), then towards the side wall (y = 0), and repeats
until nothing moves. Boxes only ever move towards the origin and never through each other,
so the result is still a valid, overlap-free packing inside the container.
"""

EPS = 1e-6
_AXES = (('oz', 'dz'), ('ox', 'dx'), ('oy', 'dy'))
_OTHER = {
    'oz': (('ox', 'dx'), ('oy', 'dy')),
    'ox': (('oy', 'dy'), ('oz', 'dz')),
    'oy': (('ox', 'dx'), ('oz', 'dz')),
}


def _overlaps(a, b, axis):
    """True if a and b overlap on both axes other than `axis` (touching does not count)."""
    for o, d in _OTHER[axis]:
        if not (a[o] < b[o] + b[d] - EPS and b[o] < a[o] + a[d] - EPS):
            return False
    return True


def _slide_limit(box, boxes, axis, dim):
    """Lowest coordinate `box` can reach along `axis` without passing through another box."""
    limit = 0.0
    start = box[axis]
    for other in boxes:
        if other is box:
            continue
        end = other[axis] + other[dim]
        if end <= start + EPS and end > limit and _overlaps(box, other, axis):
            limit = end
    return limit


def compact_boxes(boxes, max_passes=30):
    """Slides boxes (dicts with ox, oy, oz, dx, dy, dz) towards the origin in place."""
    for _ in range(max_passes):
        moved = False
        for axis, dim in _AXES:
            for box in sorted(boxes, key=lambda b: b[axis]):
                target = _slide_limit(box, boxes, axis, dim)
                if target < box[axis] - EPS:
                    box[axis] = round(target, 9)
                    moved = True
        if not moved:
            break
    return boxes


def compact_solution(solution):
    """Compacts an hga-style result ({metrics, visualization_data}) in place."""
    viz = solution.get('visualization_data') or {}
    boxes = [b for layer in viz.get('layers', []) for b in layer.get('boxes', [])]
    if not boxes:
        return solution
    compact_boxes(boxes)
    dims = viz.get('container_dims') or {}
    metrics = solution.get('metrics')
    if metrics is not None and 'dc' in dims:
        used_depth = max(b['ox'] + b['dx'] for b in boxes)
        metrics['remaining_container_depth'] = round(dims['dc'] - used_depth, 4)
    return solution
