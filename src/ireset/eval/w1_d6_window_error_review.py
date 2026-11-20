"""Compute the frozen W1 window review from saved errors and saved FK points."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from ireset.eval import w1_d6_window_error_archive as ar
from ireset.eval import w1_d6_window_error_sources as sources


def array_record(array):
    raw = array.tobytes(order='C')
    return {**ar.record(raw), 'shape': list(array.shape), 'dtype': array.dtype.str}


def bind_d6_payload(d6):
    facts = []
    for slot, (inputs, teacher, payload) in enumerate(zip(d6.inputs, d6.teachers, d6.payload, strict=True)):
        point = f'D6:payload:w{slot}:{payload["window_id"]}'
        fields = {key: array_record(val) for key, val in inputs.items() if key not in ('clip_idx', 'start')}
        ar.equal([inputs['clip_idx'].item(), inputs['start'].item()], [slot, 0], f'{point}:clip_idx_start')
        ar.equal(fields, payload['fields'], f'{point}:input')
        target = array_record(teacher['raw_x0_tgt'])
        ar.equal(target, payload['fields']['x0'], f'{point}:teacher_raw_x0_tgt')
        facts.append(dict(window_id=payload['window_id'], input=fields, teacher_raw_x0_tgt=target, clip_idx=slot, start=0))
    return facts


def bind_common(d6, b, d6_sources, b_sources):
    for key in ('joint_names', 'site_names', 'joint_range_rad', 'joint_qpos_addresses', 'joint_encoded_columns'):
        ar.require(d6_sources[key] is not None and b_sources[key] is not None, f'common:{key}:unbound')
        ar.equal(d6_sources[key], b_sources[key], f'common:{key}')
    windows = b.report['materialization']['windows']
    ar.require(isinstance(windows, list) and len(windows) == 1, 'B:materialization_window_count')
    ar.equal(windows[0]['window_id'], ar.WINDOWS[0], 'B:materialization_window')
    expected = windows[0]['x0_sha256']
    ar.equal(array_record(b.inputs['x0'])['sha256'], expected, 'B:materialization_input_x0')
    ar.equal(array_record(b.teacher['raw_x0_tgt'])['sha256'], expected, 'B:materialization_teacher_x0')
    for prefix, left, right in (('input', d6.inputs[0], b.inputs), ('teacher', d6.teachers[0], b.teacher)):
        for key in sorted(right):
            value, target = left[key], right[key]
            ar.require(value.shape == target.shape and value.dtype.str == target.dtype.str
                       and value.tobytes(order='C') == target.tobytes(order='C'), f'common:{prefix}:{key}:bits')
    return dict(window_id=ar.WINDOWS[0], x0_sha256=expected,
                inputs={key: array_record(value) for key, value in b.inputs.items()},
                teacher={key: array_record(value) for key, value in b.teacher.items()})


def window_metrics(saved, teacher, point):
    fk = np.linalg.norm(saved['fk_pred'].astype(np.float64) - teacher['fk_tgt'].astype(np.float64), axis=-1)
    ar.require(fk.tobytes(order='C') == saved['err_fk_ts'].tobytes(order='C'), f'{point}:err_fk_ts:saved_point_distance')
    metrics, spike = {}, {}
    for metric in ar.METRICS:
        errors = saved[metric.member]
        metrics[metric.name] = float(np.sqrt(np.mean(errors ** 2))) if metric.squared else float(np.mean(errors))
        index = int(np.argmax(errors.reshape(-1)))
        spike[metric.spike] = {'p95': float(np.quantile(errors, .95, method='linear')),
                               'max': float(errors.reshape(-1)[index]),
                               'worst_coordinate': [int(i) for i in np.unravel_index(index, errors.shape)]}
    ar.require(all(np.isfinite(val) for val in metrics.values()), f'{point}:metric_nonfinite')
    gates = {metric.name: metrics[metric.name] <= metric.gate for metric in ar.METRICS}
    return dict(metrics, gates=gates, window_mean_fit_passed=all(gates.values()), spike=spike)


def _trend(d, u):
    if d <= -.10:
        return 'down'
    if d >= .10:
        return 'up'
    return 'flat' if u <= .20 else 'volatile'


def interval_direction(delta_a, delta_b):
    if abs(delta_a) < .05 or abs(delta_b) < .05:
        return 'small'
    return 'same' if (delta_a > 0.) == (delta_b > 0.) else 'opposite'


def learning(curves):
    summaries, pairs = [], {}
    for metric in ar.METRICS:
        key = metric.name
        ratios = []
        for slot, window in enumerate(ar.WINDOWS):
            rows = curves[slot::len(ar.WINDOWS)]
            values = [row['metrics'][key] for row in rows]
            tail_si = values[80:]
            ratio = [row['r'][key] for row in rows[80:]]
            ratios.append(ratio)
            a, b = float(np.median(ratio[:5])), float(np.median(ratio[-5:]))
            scale = max(1., a)
            d = (b - a) / scale
            p10, p90 = np.quantile(ratio, [.10, .90], method='linear').tolist()
            u = (p90 - p10) / scale
            tail_min, tail_max = min(tail_si), max(tail_si)
            summaries.append(dict(window_id=window, metric=key, unit=metric.unit,
                                  delta_500_1000_si=values[100] - values[50],
                                  tail=dict(n=21, median_si=float(np.median(tail_si)),
                                            min_si=tail_min, min_step=800 + 10 * tail_si.index(tail_min),
                                            max_si=tail_max, max_step=800 + 10 * tail_si.index(tail_max),
                                            passing=sum(row['metrics']['gates'][key] for row in rows[80:])),
                                  a=a, b=b, s=scale, d=d, u=u, trend=_trend(d, u)))
        pairs[key] = []
        for i, j in combinations(range(len(ar.WINDOWS)), 2):
            categories = [interval_direction(ratios[i][k + 1] - ratios[i][k], ratios[j][k + 1] - ratios[j][k])
                          for k in range(20)]
            pairs[key].append(dict(windows=[ar.WINDOWS[i], ar.WINDOWS[j]],
                                   **{kind: categories.count(kind) for kind in ('same', 'opposite', 'small')}))
    return dict(summaries=summaries, co_movement=pairs)


def contributions(errors, metric, axis):
    mass = errors ** 2 if metric.squared else errors
    if errors.ndim == 1:
        numerators = mass
    else:
        numerators = np.sum(mass, axis=1 if axis == 'frame' else 0)
    total = float(np.sum(mass))
    shares = None if total == 0. else (numerators / total).tolist()
    ar.require(shares is None or abs(sum(shares) - 1.) <= 1e-12, f'{metric.name}:{axis}:shares_sum')
    return dict(numerators=numerators.tolist(), denominator=total, shares=shares,
                max_index=None if shares is None else int(np.argmax(numerators)))


def fixed_group(samples, axis, size):
    anchor = samples[800]
    selected = (sorted(range(len(anchor['numerators'])), key=lambda i: (-anchor['numerators'][i], i))[:size]
                if anchor['denominator'] != 0. else None)
    end = samples[1000]
    points, hits = [], 0
    if selected is not None:
        for step in range(800, 1001, 10):
            sample = samples[step]
            vals = [sample['numerators'][i] for i in selected]
            numerator = sum(vals)
            denominator = sample['denominator']
            share = numerator / denominator if denominator != 0. else None
            points.append(dict(step=step, numerator=numerator, denominator=denominator, share=share))
            hits += int(step > 800 and share is not None and share >= .5)
            if axis == 'site':
                points[-1]['members'] = [dict(index=index, mean_m=mass / 60., numerator=mass,
                                              share=mass / denominator if denominator != 0. else None)
                                         for index, mass in zip(selected, vals, strict=True)]
    last = points[-1]['share'] if points else None
    return dict(selected=selected, points=points, hits=hits, n_tail=20, last_share=last,
                persistent=hits >= 16 and last is not None and last >= .5,
                outside_at_1000=[i for i in range(len(end['numerators'])) if i not in (selected or [])])


def localization(samples, curves, names):
    windows = {}
    for slot, window in enumerate(ar.WINDOWS):
        terminal = curves[-len(ar.WINDOWS) + slot]['metrics']
        reports = {}
        for metric in ar.METRICS:
            axes = {}
            for axis, size in metric.axes:
                points = samples[window][metric.name][axis]
                axes[axis] = dict(snapshots={str(t): points[t] for t in (500, 800, 1000)},
                                  fixed_group=fixed_group(points, axis, size))
            reports[metric.name] = dict(axes=axes, step1000_si=terminal[metric.name], spike=terminal['spike'][metric.spike])
        windows[window] = reports
    return dict(joint_names=names['joint_names'], site_names=names['site_names'], windows=windows,
                spike_gate_status='not_evaluated')


def first_run(steps, delta, direction):
    for idx in range(len(delta) - 2):
        segment = delta[idx:idx+3]
        if all(direction * value >= .25 for value in segment):
            return dict(steps=list(steps[idx:idx+3]), delta_r=segment, starts_at_zero=steps[idx] == 0)
    return None


def pair_curves(curves, values):
    rows = []
    for curve, b in zip(curves[::len(ar.WINDOWS)], values, strict=True):
        metrics = {}
        for metric in ar.METRICS:
            d6, other = curve['metrics'][metric.name], b[metric.name]
            ratio_b = other / metric.gate
            metrics[metric.name] = dict(D6=d6, B=other, D6_minus_B=d6 - other,
                                        r_D6=curve['r'][metric.name], r_B=ratio_b,
                                        delta_r=curve['r'][metric.name] - ratio_b)
        rows.append(dict(window_id=ar.WINDOWS[0], step=curve['step'], mark=curve['mark'], metrics=metrics))
    first = {}
    for metric in ar.METRICS:
        delta = [row['metrics'][metric.name]['delta_r'] for row in rows]
        first[metric.name] = dict(positive=first_run(ar.STEPS, delta, 1), negative=first_run(ar.STEPS, delta, -1))
    return rows, first


def choose_candidate(loc, comparable):
    counts = defaultdict(list)
    for window, descriptions in loc['windows'].items():
        group = descriptions[ar.FK_METRIC]['axes']['site']['fixed_group']
        if group['selected'] is not None:
            eligible = counts[tuple(sorted(group['selected']))]
            if group['persistent']:
                eligible.append(window)
    groups = [dict(indices=list(indices), window_ids=members, count=len(members))
              for indices, members in sorted(counts.items())]
    candidate = next((group for group in groups if comparable and group['count'] >= 4), None)
    decision = 'group_fk_weight_diagnostic_candidate' if candidate else 'attribution_insufficient'
    reason = ('same_site_set_persistent_in_at_least_four_windows' if candidate else
              'common_window_not_comparable' if not comparable else 'no_same_site_set_persistent_in_four_windows')
    return dict(decision=decision, reason=reason, candidate=candidate, site_group_counts=groups)


def analyze(reader, progress, *, blob=sources.git_blob):
    d6 = ar.open_d6(reader)
    d6_sources = sources.review_sources(reader, 'D6', d6.launch['implementation']['files'], blob=blob)
    slot_facts = bind_d6_payload(d6)
    curves = []
    samples = {window: {m.name: {axis: {} for axis, _ in m.axes} for m in ar.METRICS} for window in ar.WINDOWS}
    for parent, windows in d6.probes():
        step = parent['step']
        for slot, (saved, original) in enumerate(zip(windows, parent['windows'], strict=True)):
            point = f'D6:step{step}:slot{slot}:{ar.WINDOWS[slot]}'
            metrics = window_metrics(saved, d6.teachers[slot], point)
            ar.equal(metrics, original['metrics'], f'{point}:metrics')
            ar.equal(original['spike_gate_status'], 'not_evaluated', f'{point}:spike_gate_status')
            curves.append(dict(step=step, window_id=ar.WINDOWS[slot], batch_pos=slot, metrics=metrics,
                               mark=ar.MARKS.get(step), r={m.name: metrics[m.name] / m.gate for m in ar.METRICS}))
            if step == 500 or step >= 800:
                for m in ar.METRICS:
                    for axis, _ in m.axes:
                        samples[ar.WINDOWS[slot]][m.name][axis][step] = contributions(saved[m.member], m, axis)
            progress['d6_window_rows'] = len(curves)
    final = [{key: row[key] for key in ('step', 'window_id', 'batch_pos', 'metrics')} for row in curves[-6:]]
    ar.equal(final, d6.review['final']['final_windows'], 'D6:independent_historical_final_rows')
    ar.equal(final, d6.report['final']['final_windows'], 'D6:local_historical_final_rows')
    counts = [sum(row['metrics']['gates'][m.name] for row in final) for m in ar.METRICS]
    conjunct = sum(row['metrics']['window_mean_fit_passed'] for row in final)
    ar.equal([counts, conjunct, d6.review['final']['n_passing_windows']], [[3, 0, 0, 0], 0, 0], 'D6:final_denominator')
    progress['d6_final_rows_verified'] = True
    learning_output = learning(curves)
    local = localization(samples, curves, d6_sources)
    pair_b, binding = None, None
    try:
        b = ar.open_b(reader)
        pair_b = sources.review_sources(reader, 'B', b.admission['package']['code_identity']['implementation_files'], blob=blob)
        binding = bind_common(d6, b, d6_sources, pair_b)
    except ar.INVALID_ARCHIVE as exc:
        paired = dict(status='not_comparable', problem=str(exc), rows=None, first=None)
    else:
        rows, first = pair_curves(curves, b.metrics)
        paired = dict(status='comparable', problem=None, rows=rows, first=first)
        progress['paired_rows'] = len(rows)
    decision = choose_candidate(local, paired['status'] == 'comparable')
    return dict(curves=curves, learning=learning_output, localization=local, paired=paired,
                decision=decision, final=dict(windows=final, counts=counts, conjunction=conjunct),
                binding=dict(d6_sources=d6_sources, b_sources=pair_b, d6_slots=slot_facts, common=binding))
