"""Independent numerical consumer for the W1 review's saved summaries.

Only file format admission, constants and the static source inspection are
shared with the producer. All numerical products are checked from raw input.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from statistics import median

import numpy as np

from ireset.eval import w1_d6_window_error_archive as ar
from ireset.eval import w1_d6_window_error_sources as sources


def _array_meta(array):
    return dict(ar.record(memoryview(np.asarray(array, order='C')).cast('B')),
                shape=list(array.shape), dtype=array.dtype.str)


def _d6_slots(archive):
    records = []
    for slot, row in enumerate(archive.payload):
        inputs, teacher = archive.inputs[slot], archive.teachers[slot]
        point = f'verify:D6:w{slot}'
        fields = {name: _array_meta(inputs[name]) for name in row['fields']}
        ar.equal(sorted(fields), sorted(set(ar.PROBE_INPUTS) - {'clip_idx', 'start'}), f'{point}:fields')
        ar.equal(fields, row['fields'], f'{point}:payload_inputs')
        t = _array_meta(teacher['raw_x0_tgt'])
        ar.equal(t, row['fields']['x0'], f'{point}:teacher_x0')
        ar.equal(inputs['clip_idx'].item(), slot, f'{point}:clip_idx')
        ar.equal(inputs['start'].item(), 0, f'{point}:start')
        records.append(dict(window_id=row['window_id'], input=fields, teacher_raw_x0_tgt=t, clip_idx=slot, start=0))
    return records


def _check_pair_inputs(d6, b, d6_src, b_src):
    for label in ('joint_names', 'site_names', 'joint_range_rad', 'joint_qpos_addresses', 'joint_encoded_columns'):
        ar.require(b_src[label] is not None and d6_src[label] is not None, f'common:{label}:unbound')
        ar.equal(b_src[label], d6_src[label], f'common:{label}')
    materialized = b.report['materialization']['windows']
    ar.require(isinstance(materialized, list) and len(materialized) == 1, 'B:materialization_window_count')
    mat, = materialized
    ar.equal(mat['window_id'], ar.WINDOWS[0], 'B:materialization_window')
    mat_sha = mat['x0_sha256']
    groups = [('input', ar.COMMON_INPUTS, d6.inputs[0], b.inputs, 'x0'),
              ('teacher', ar.TEACHER, d6.teachers[0], b.teacher, 'raw_x0_tgt')]
    fact = dict(window_id=ar.WINDOWS[0], x0_sha256=mat_sha)
    for name, _, _, right, x0 in groups:
        ar.equal(_array_meta(right[x0])['sha256'], mat_sha, f'B:materialization_{name}_x0')
    for name, keys, left, right, _ in groups:
        for key in sorted(keys):
            a, b_value = np.asarray(left[key], order='C'), np.asarray(right[key], order='C')
            ar.require(a.shape == b_value.shape and a.dtype.str == b_value.dtype.str
                       and np.array_equal(a.view(np.uint8), b_value.view(np.uint8)), f'common:{name}:{key}:bits')
        fact['inputs' if name == 'input' else name] = {key: _array_meta(right[key]) for key in keys}
    return fact


def _reduce(arrays, teacher, point):
    diff = np.subtract(arrays['fk_pred'], teacher['fk_tgt'], dtype=np.float64)
    distance = np.sqrt(np.add.reduce(np.square(diff), axis=-1))
    ar.require(np.array_equal(distance.view(np.uint64), arrays['err_fk_ts'].view(np.uint64)), f'{point}:saved_fk_distance')
    values, gates, spikes = {}, {}, {}
    for m in ar.METRICS:
        e = arrays[m.member]
        values[m.name] = float(np.sqrt(np.square(e).mean()) if m.squared else e.mean())
        gates[m.name] = values[m.name] <= m.gate
        flat = e.flatten().tolist()
        peak = max(flat)
        first = flat.index(peak)
        coord = [first] if e.ndim == 1 else list(divmod(first, e.shape[1]))
        spikes[m.spike] = dict(p95=float(np.percentile(e, 95, method='linear')), max=peak, worst_coordinate=coord)
    return dict(values, gates=gates, window_mean_fit_passed=all(gates.values()), spike=spikes)


def _parts(matrix, metric, axis):
    data = np.square(matrix) if metric.squared else matrix
    vector = (data if data.ndim == 1 else
              np.add.reduce(data, axis=0 if axis != 'frame' else -1))
    den = float(np.add.reduce(data, axis=None))
    if den:
        proportions = np.divide(vector, den).tolist()
        ar.require(abs(sum(proportions) - 1.) <= 1e-12, f'verify:{metric.name}:{axis}:sum')
        max_index = vector.tolist().index(float(vector.max()))
    else:
        proportions, max_index = None, None
    return dict(numerators=vector.tolist(), denominator=den, shares=proportions, max_index=max_index)


def _verify_learning(product, by_window):
    for k, m in enumerate(ar.METRICS):
        series = {}
        for i, id_ in enumerate(ar.WINDOWS):
            points = by_window[id_]
            tail = [(row['step'], row['metrics'][m.name]) for row in points if row['step'] >= 800]
            values = [value for _, value in tail]
            r = [val / m.gate for val in values]
            series[id_] = r
            a, b = median(r[:5]), median(r[-5:])
            s = 1.0 if a < 1.0 else a
            d = (b - a) / s
            # Twenty intervals make linear P10/P90 exact order statistics.
            ordered = sorted(r)
            u = (ordered[18] - ordered[2]) / s
            label = ('up' if d >= .10 else 'down' if d <= -.10 else
                     'volatile' if u > .20 else 'flat')
            low = min(tail, key=lambda row: (row[1], row[0]))
            high = min(tail, key=lambda row: (-row[1], row[0]))
            summary = dict(window_id=id_, metric=m.name, unit=m.unit,
                           delta_500_1000_si=points[-1]['metrics'][m.name] - points[50]['metrics'][m.name],
                           tail=dict(n=21, median_si=median(values), min_si=low[1], min_step=low[0],
                                     max_si=high[1], max_step=high[0], passing=sum(value <= m.gate for value in values)),
                           a=a, b=b, s=s, d=d, u=u, trend=label)
            ar.equal(product['summaries'][k * len(ar.WINDOWS) + i], summary, f'verify:learning:{id_}:{m.name}')
        diffs = {win: np.diff(r) for win, r in series.items()}
        pairs = []
        for x, y in combinations(ar.WINDOWS, 2):
            dx, dy = diffs[x], diffs[y]
            admitted = (np.abs(dx) >= .05) & (np.abs(dy) >= .05)
            opposite = np.signbit(dx) != np.signbit(dy)
            pairs.append(dict(windows=[x, y], same=int((admitted & ~opposite).sum()),
                              opposite=int((admitted & opposite).sum()), small=int((~admitted).sum())))
        ar.equal(product['co_movement'][m.name], pairs, f'verify:co_movement:{m.name}')
    ar.equal(len(product['summaries']), 24, 'verify:learning_count')


def _locate_product(product, windows, contributions, src):
    ar.equal(product['site_names'], src['site_names'], 'verify:localization_sites')
    ar.equal(product['joint_names'], src['joint_names'], 'verify:localization_joints')
    ar.equal(product['spike_gate_status'], 'not_evaluated', 'verify:localization_spike_policy')
    groups = {}
    for win, final in windows.items():
        for m in ar.METRICS:
            actual = product['windows'][win][m.name]
            ar.equal([actual['step1000_si'], actual['spike']], [final[m.name], final['spike'][m.spike]], f'verify:localization:{win}:{m.name}:end')
            for axis, k in m.axes:
                point = f'verify:localization:{win}:{m.name}:{axis}'
                samples = contributions[win, m.name, axis]
                ar.equal(actual['axes'][axis]['snapshots'], {str(t): samples[t] for t in (500, 800, 1000)}, f'{point}:snapshots')
                anchor = samples[800]
                rank = np.argsort(-np.array(anchor['numerators']), kind='stable')
                indices = rank[:k].tolist() if anchor['denominator'] else None
                group_points = []
                for step in range(800, 1001, 10) if indices is not None else ():
                    values = samples[step]['numerators']
                    denom = samples[step]['denominator']
                    numerator = sum(values[index] for index in indices)
                    row = dict(step=step, numerator=numerator, denominator=denom,
                               share=numerator / denom if denom else None)
                    if axis == 'site':
                        row['members'] = [dict(index=i, mean_m=values[i] / 60., numerator=values[i],
                                               share=values[i] / denom if denom else None) for i in indices]
                    group_points.append(row)
                hits = len([p for p in group_points[1:] if p['share'] is not None and p['share'] >= .5])
                last = group_points[-1]['share'] if group_points else None
                persistent = bool(last is not None and last >= .5 and hits >= 16)
                expected = dict(selected=indices, points=group_points, hits=hits, n_tail=20, last_share=last,
                                persistent=persistent,
                                outside_at_1000=sorted(set(range(len(rank))) - set(indices or [])))
                ar.equal(actual['axes'][axis]['fixed_group'], expected, f'{point}:fixed_group')
                if axis == 'site' and indices is not None:
                    matches = groups.setdefault(tuple(sorted(indices)), [])
                    if persistent:
                        matches.append(win)
    return groups


def _first_by_scan(pairs, key, positive):
    consecutive = []
    for row in pairs:
        value = row['metrics'][key]['delta_r']
        meets = value >= .25 if positive else value <= -.25
        if not meets:
            consecutive.clear()
            continue
        consecutive.append((row['step'], value))
        if len(consecutive) == 3:
            return dict(steps=[s for s, _ in consecutive], delta_r=[v for _, v in consecutive], starts_at_zero=consecutive[0][0] == 0)
    return None


def verify(reader, product, *, blob=sources.git_blob, progress=None):
    archive = ar.open_d6(reader)
    src = sources.review_sources(reader, 'D6', archive.launch['implementation']['files'], blob=blob)
    ar.equal(product['binding']['d6_sources'], src, 'verify:D6:source_identity')
    ar.equal(product['binding']['d6_slots'], _d6_slots(archive), 'verify:D6:payload_slots')
    row_count = 0
    series = {key: [] for key in ar.WINDOWS}
    contributions = {}
    for parent, snapshots in archive.probes():
        step = parent['step']
        for slot, win in enumerate(ar.WINDOWS):
            point = f'verify:step{step}:{win}'
            state = snapshots[slot]
            m = _reduce(state, archive.teachers[slot], point)
            ar.equal(m, parent['windows'][slot]['metrics'], f'{point}:archive_metrics')
            ar.equal(parent['windows'][slot]['spike_gate_status'], 'not_evaluated', f'{point}:spike_status')
            row = dict(step=step, window_id=win, batch_pos=slot, metrics=m, mark=ar.MARKS.get(step),
                       r={spec.name: m[spec.name] / spec.gate for spec in ar.METRICS})
            ar.equal(product['curves'][row_count], row, f'{point}:published_curve')
            series[win].append(row)
            row_count += 1
            if progress is not None:
                progress['independent_d6_window_rows'] = row_count
            if step == 500 or step >= 800:
                for spec in ar.METRICS:
                    for axis, _ in spec.axes:
                        contributions.setdefault((win, spec.name, axis), {})[step] = _parts(state[spec.member], spec, axis)
    ar.equal([row_count, len(product['curves'])], [606, 606], 'verify:curve_count')
    final = [{key: rows[-1][key] for key in ('batch_pos', 'window_id', 'step', 'metrics')} for rows in series.values()]
    ar.equal(final, archive.review['final']['final_windows'], 'verify:D6:historical_final')
    ar.equal(final, archive.report['final']['final_windows'], 'verify:D6:report_final')
    wins_by_gate = Counter(key for row in final for key, passed in row['metrics']['gates'].items() if passed)
    counts = [wins_by_gate[m.name] for m in ar.METRICS]
    all_pass = sum(row['metrics']['window_mean_fit_passed'] for row in final)
    ar.equal([counts, all_pass], [[3, 0, 0, 0], 0], 'verify:gate_counts')
    ar.equal(product['final'], dict(windows=final, counts=counts, conjunction=all_pass), 'verify:published_final')
    _verify_learning(product['learning'], series)
    groups = _locate_product(product['localization'], {w: rows[-1]['metrics'] for w, rows in series.items()}, contributions, src)

    b_src = None
    try:
        b = ar.open_b(reader)
        b_src = sources.review_sources(reader, 'B', b.admission['package']['code_identity']['implementation_files'], blob=blob)
        common = _check_pair_inputs(archive, b, src, b_src)
    except ar.INVALID_ARCHIVE as exc:
        ar.equal(product['paired']['status'], 'not_comparable', 'verify:pair_ineligible')
        ar.equal([product['paired']['rows'], product['paired']['first'], product['binding']['common']],
                 [None, None, None], 'verify:pair_ineligible_suppression')
        ar.equal(product['paired']['problem'], str(exc), 'verify:pair_ineligible_reason')
        pair_count, pair_problem = 0, str(exc)
    else:
        ar.equal(product['binding']['common'], common, 'verify:bound_pair_inputs')
        ar.equal(product['paired']['status'], 'comparable', 'verify:pair_eligible')
        ar.equal(product['paired']['problem'], None, 'verify:pair_eligible_problem')
        rows = []
        for d6, other in zip(series[ar.WINDOWS[0]], b.metrics, strict=True):
            metrics = {}
            for spec in ar.METRICS:
                a, v = d6['metrics'][spec.name], other[spec.name]
                a_ratio, b_ratio = a / spec.gate, v / spec.gate
                metrics[spec.name] = dict(D6=a, B=v, D6_minus_B=a - v, delta_r=a_ratio - b_ratio, r_D6=a_ratio, r_B=b_ratio)
            rows.append(dict(window_id=ar.WINDOWS[0], step=d6['step'], mark=ar.MARKS.get(d6['step']), metrics=metrics))
        ar.equal(product['paired']['rows'], rows, 'verify:paired_rows')
        runs = {spec.name: dict(positive=_first_by_scan(rows, spec.name, True), negative=_first_by_scan(rows, spec.name, False))
                for spec in ar.METRICS}
        ar.equal(product['paired']['first'], runs, 'verify:paired_first_runs')
        pair_count, pair_problem = len(rows), None
    ar.equal(product['binding']['b_sources'], b_src, 'verify:B:source_identity')
    group_counts = [dict(indices=list(key), window_ids=matches, count=len(matches)) for key, matches in sorted(groups.items())]
    qualified = [g for g in group_counts if g['count'] >= 4] if pair_count else []
    ar.require(len(qualified) <= 1, 'verify:ambiguous_candidate')
    if qualified:
        decision, reason = 'group_fk_weight_diagnostic_candidate', 'same_site_set_persistent_in_at_least_four_windows'
    else:
        decision = 'attribution_insufficient'
        reason = 'no_same_site_set_persistent_in_four_windows' if pair_count else 'common_window_not_comparable'
    ar.equal(product['decision'], dict(decision=decision, reason=reason, candidate=qualified[0] if qualified else None,
                                       site_group_counts=group_counts), 'verify:decision')
    return dict(d6_window_rows=row_count, metric_reductions=row_count * 4, final_windows=6, learning_curves=24,
                window_pairs=60, localization_axes=36, paired_rows=pair_count, pair_problem=pair_problem)
