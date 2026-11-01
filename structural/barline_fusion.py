import numpy as np

def ordered_staff_trace(trace):
    """HOMR's joined anchors can overlap; interpolation requires increasing x."""
    groups = {}
    for point in trace:
        groups.setdefault(float(point['x']), []).append(point)
    return [dict(x=x, top=float(np.median([p['top'] for p in points])),
                 bottom=float(np.median([p['bottom'] for p in points])))
            for x, points in sorted(groups.items())]


