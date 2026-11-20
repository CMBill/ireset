"""Static review of each run's bound source bytes. No historical imports."""

from __future__ import annotations

import ast
import re
import subprocess
from xml.etree import ElementTree

import numpy as np

from ireset.eval import w1_d6_window_error_archive as ar


REVISIONS = {'D6': 'b138bb488b49716bb348cbddd1f90f3e683f18b4', 'B': 'e96e0adea1573eade8c943233ae56f42ec6120ce'}
G1 = 'embodied/g1_spec.py'
TOKENS = 'model/tokens.py'
FIT = 'eval/w1_smallset_fit_diagnostic.py'
TEACHER = 'data/teacher.py'
WINDOW = 'eval/w1_smallset_fit_r_admission.py'
D6_REAL = 'eval/w1_d6_joint_fit_real.py'

# Digests cover the reviewed AST nodes in listed order, with the Python 3.12
# schema and no line positions. This also checks definitions unchanged across
# versions whose entire file hashes differ. Git bytes are checked separately.
COMMON = {
    G1: ('G1Spec', '9031f4cc5637371f73c2e884e09e8c1a70833d9a46bb0dce6d722f4284a4d156'),
    TOKENS: ('StateTokenizer', '191178948cda0735b844757680049577acd6e77d30fce9dfc59b0648e76074df'),
    FIT: ('MetricDecoder.decode so3_geodesic_angles_rad _p95_linear _max_first window_spike_stats '
          'compute_window_metrics FitDriver._ensure_teacher FitDriver._run_probe',
          '919957f45ec89966d2b80d552dc28a5258eb9bd79c7dd98f12df48aaac7f4fd8'),
    'kinematics/torch_adapter.py': ('G1TorchFK.site_world_positions', 'e5494c4f3041309944d5490ccc0db48265c471550d429b6d065f288b78ec4c8f'),
    'train/losses.py': ('LossAssembler.decode_clip LossAssembler.__call__', '4364e5d15c08693cd50f70795d34c89d99d4904afc0e19ec25121474365fb3b4'),
    TEACHER: ('convert_teacher_qpos load_teacher_npz build_interaction_clip', '58ecac2423c75b1680739117e234be12f9fb6f0304e1c49ac9c5895efa111b8e'),
    WINDOW: ('_window_item', '4042107ea598a4b8a0376773299d64c89d051bab8c79dd163e8a84ef2766d86a'),
}
SIDE = {
    'D6': {
        'data/schema.py': ('InteractionClip', 'dfc1ed6ed7f0b9fcfdcbe345a50c20cbd5237f0db0d997169669ec915d5177ee'),
        D6_REAL: ('BoundAssets clip_from_snapshots apparatus build_windows run_d6',
                  '012cadf07b9bf0bf1210251ff3a5864291fb08e020760885ad7abec32052a34e'),
        'eval/w1_d6_joint_fit.py': ('JointDriver._ensure_teacher JointDriver._run_probe', '6d21855a83a346cc73e3069ff639a080e93297040b2110032274f5c63f19b0d8'),
        'eval/w1_d6_joint_fit_evidence.py': ('_verify_reduction saved_metrics', 'a676fc8ec77dea6a37a31e591c506e7c4470f8012e93e6a6b9232371cf7519bd'),
    },
    'B': {
        WINDOW: ('materialize_real_world materialize_fit_apparatus', '782d7ff1568c6eb9a8ad1365d5eb89255492675d1afd128ac9a5013961ab8b36'),
        'data/w1_training.py': ('load_w1_derived_clip', '8f7c1087389c5f26c3d09c738e6ee2970fb2f2df08a09acd4cb3ef3cf5fd42e5'),
        'eval/w1_d1_lr_workflow.py': ('run_candidate', '495069c388fa11c3fe960ecaf3b6fd72655f8e3553cbca9a16ff51699e185179'),
        'eval/w1_d1_lr_schedule.py': ('ScheduleDriver build_comparison', '4f9bf9b51b75d65930165e0bf92a4f1dae4a99ec8d372d36073904df3563d42a'),
    },
}


def historical_paths(side):
    return tuple(f'src/ireset/{name}' for name in dict.fromkeys((*COMMON, *SIDE[side])))


def git_blob(reader, revision, path):
    ar.require(any(revision == rev and path in historical_paths(side) for side, rev in REVISIONS.items()),
               f'{path}:historical_blob_not_whitelisted')
    reader.check(path)
    try:
        result = subprocess.run(['git', 'show', f'{revision}:{path}'], cwd=reader.root, capture_output=True,
                                check=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ar.Refused(f'{path}:historical_git_blob:{type(exc).__name__}') from exc
    reader.check(path)
    return result.stdout


def rehash_sources(reader, blob):
    for side, files in reader.source_receipts.items():
        for path, expected in files.items():
            ar.equal(ar.record(blob(reader, REVISIONS[side], path)), expected, f'{side}:{path}:historical_rehash')


def _node(tree, selector):
    for name in selector.split('.'):
        matches = [node for node in tree.body if getattr(node, 'name', None) == name]
        ar.require(len(matches) == 1, f'source_symbol:{selector}')
        tree = matches[0]
    return tree


def _assignment(tree, target):
    matches = [statement.value for statement in tree.body if isinstance(statement, (ast.Assign, ast.AnnAssign))
               and any(ast.unparse(node) == target for node in
                       (statement.targets if isinstance(statement, ast.Assign) else [statement.target]))]
    ar.require(len(matches) == 1, f'source_assignment:{target}')
    return matches[0]


def axis_sentinels(trees, side, ranges):
    """Follow only reviewed AST slices and identities with 29 unique columns.

    The values exercise the static contract. No historical function is called.
    The control uses the actual slices found in the bound syntax tree.
    """
    expected = np.arange(29)
    source = np.arange(43) - 7
    pack = np.full(43, -999)
    converter = _node(trees[TEACHER], 'convert_teacher_qpos')
    qpos_slice = _assignment(converter, 'out[7:36]')

    def apply_subscript(node, values, name, point):
        ar.require(isinstance(node, ast.Subscript) and ast.unparse(node.value) == name, point)
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        parts = []
        for part in indices:
            ar.require(isinstance(part, ast.Slice), point)
            try:
                parts.append(slice(*(ast.literal_eval(bound) if bound is not None else None
                                     for bound in (part.lower, part.upper, part.step))))
            except ValueError as exc:
                raise ar.Refused(point) from exc
        return values[tuple(parts)]

    converted = apply_subscript(qpos_slice, source, 'row', 'joint_axis:qpos_slice')
    ar.require(np.array_equal(converted, expected), 'joint_axis:qpos_slice')
    pack[7:36] = converted
    if side == 'D6':
        builder = _node(trees[D6_REAL], 'clip_from_snapshots')
        q = _assignment(builder, '(root, q, box)').elts[1]
    else:
        builder = _node(trees[TEACHER], 'build_interaction_clip')
        q = _assignment(builder, 'q')
    ar.require(isinstance(q, ast.Call) and isinstance(q.func, ast.Attribute) and q.func.attr == 'copy', 'joint_axis:scene_slice')
    sliced = apply_subscript(q.func.value, pack[None, :], 'scene', 'joint_axis:scene_slice')
    ar.require(np.array_equal(sliced[0], expected), 'joint_axis:scene_slice')
    clip_return = next(node.value for node in builder.body if isinstance(node, ast.Return))
    clip_q = [ast.unparse(k.value) for k in clip_return.keywords if k.arg == 'q']
    ar.equal(clip_q, ['q'], 'joint_axis:clip_q')
    window = _node(trees[WINDOW], '_window_item')
    ar.equal(ast.unparse(_assignment(window, '(root, q)').elts[1]), 'clip.q[start:start + T_WINDOW]', 'joint_axis:window_q')
    ar.equal(ast.unparse(_assignment(window, 'x0')), 'tokenizer.encode(root, q, cond.world_from_clip)', 'joint_axis:tokenizer_q')

    tokenizer = trees[TOKENS]
    encoder = _node(tokenizer, 'StateTokenizer.encode')
    concat = next(node.value for node in encoder.body if isinstance(node, ast.Return))
    ar.equal(ast.unparse(concat), 'np.concatenate([pos_n, r6, self.joints_to_u(q)], axis=1)', 'joint_axis:encode_columns')
    # Distinct normalized values in the interior of each named joint range.
    marker = (sliced + 1.) / 32. - .5
    lo, hi = ranges.T
    angles = lo + (marker + 1.) * (hi - lo) / 2.
    tokens = np.concatenate([np.zeros((1, 3)), np.zeros((1, 6)), 2. * (angles - lo) / (hi - lo) - 1.], axis=1)
    decoder = _node(tokenizer, 'StateTokenizer.decode')
    call = _assignment(decoder, 'q')
    ar.require(isinstance(call, ast.Call) and ast.unparse(call.func) == 'self.u_to_joints' and len(call.args) == 1,
               'joint_axis:decode_columns')
    decoded_u = apply_subscript(call.args[0], tokens, 'x0', 'joint_axis:decode_columns')
    ar.require(np.allclose(decoded_u, marker, rtol=0., atol=1e-14), 'joint_axis:decode_columns')
    decoder = _node(trees[FIT], 'MetricDecoder.decode')
    clip = _assignment(decoder, 'u')
    ar.require(isinstance(clip, ast.Call) and ast.unparse(clip.func) == 'np.clip', 'joint_axis:metric_decode')
    decoded_u = apply_subscript(clip.args[0], tokens, 'x', 'joint_axis:metric_decode')
    ar.require(np.allclose(decoded_u, marker, rtol=0., atol=1e-14), 'joint_axis:metric_decode')


def locked_joint_table(raw):
    root = ElementTree.fromstring(raw)
    ar.require(not root.findall('.//include') and root.find('compiler').get('angle') == 'radian', 'locked_g1:xml_layout')
    world = root.find('worldbody')
    joints = [node for node in world.iter() if node.tag in ('joint', 'freejoint')]
    ar.require(len(joints) == 30 and (joints[0].tag == 'freejoint' or joints[0].get('type') == 'free'), 'locked_g1:qpos_free_base')
    actuated = joints[1:]
    ar.require(all(node.tag == 'joint' and node.get('type', 'hinge') == 'hinge' for node in actuated), 'locked_g1:joint_types')
    names = [node.get('name') for node in actuated]
    ranges = np.array([[float(value) for value in node.get('range', '').split()] for node in actuated])
    ar.require(ranges.shape == (29, 2) and np.isfinite(ranges).all() and (ranges[:, 1] > ranges[:, 0]).all(), 'locked_g1:joint_ranges')
    return names, ranges, [site.get('name') for site in world.iter('site')]


def review_sources(reader, side, files, *, blob=git_blob):
    trees, evidence = {}, {}
    revision = REVISIONS[side]
    for rel in historical_paths(side):
        expected = files.get(rel)
        if expected is None and rel == f'src/ireset/{G1}' and side == 'D6':
            evidence[rel] = {'status': 'unbound'}
            continue
        source_sha = expected.get('sha256') if isinstance(expected, dict) else expected
        ar.require(isinstance(source_sha, str) and re.fullmatch(r'[0-9a-f]{64}', source_sha) is not None,
                   f'{side}:{rel}:historical_source_unbound')
        raw = blob(reader, revision, rel)
        rec = ar.record(raw)
        reader.source_receipts.setdefault(side, {})[rel] = rec
        ar.equal(rec['sha256'], source_sha, f'{side}:{rel}:historical_sha')
        if side == 'D6':
            ar.equal(rec, expected, f'{side}:{rel}:historical_length')
        evidence[rel] = rec
        trees[rel.removeprefix('src/ireset/')] = ast.parse(raw)

    # The model is inspected as binding metadata only. Traversal counts a
    # seven-coordinate free joint followed by the 29 one-coordinate hinges.
    names, ranges, xml_sites = locked_joint_table(reader.read(ar.LOCKED_G1))
    axis_sentinels(trees, side, ranges)
    for dictionary in (COMMON, SIDE[side]):
        for rel, (symbols, expected) in dictionary.items():
            if rel in trees:
                ast_bytes = '\n'.join(ast.dump(_node(trees[rel], name)) for name in symbols.split()).encode()
                ar.equal(ar.record(ast_bytes)['sha256'], expected, f'{side}:{rel}:audited_symbols:{symbols}')
    gates = ('GATE_ROOT_POS_RMSE_M', 'GATE_ROOT_ROT_MEAN_RAD', 'GATE_JOINT_RMSE_RAD', 'GATE_FK_MPJPE_M')
    for name, metric in zip(gates, ar.METRICS, strict=True):
        value = _assignment(trees[FIT], name)
        if name == 'GATE_ROOT_ROT_MEAN_RAD':
            ar.equal(ast.unparse(value), 'math.pi / 180.0', f'{side}:rotation_gate')
        else:
            ar.equal(ast.literal_eval(value), metric.gate, f'{side}:{name}')
    joint_names = site_names = None
    if G1 in trees:
        spec = trees[G1]
        ar.equal(ast.unparse(_assignment(spec, 'JOINT_QPOS_SLICE')), 'slice(7, 36)', f'{side}:joint_qpos_slice')
        joint_names = list(ast.literal_eval(_assignment(spec, 'JOINT_NAMES')))
        site_names = list(ast.literal_eval(_assignment(spec, 'KPT_TO_SMPLH')))
        ar.equal(joint_names, names, f'{side}:locked_mjcf_joint_order')
        ar.require(len(site_names) == 15 and len(set(site_names)) == 15, f'{side}:site_roster')
        ar.require(all(xml_sites.count(site) == 1 for site in site_names), f'{side}:locked_mjcf_site_roster')
        ar.equal(evidence[f'src/ireset/{G1}']['sha256'],
                 'c4bf6946803c81025f1ea385d5345ac4c60f81a47921d9404a048e4c25ad7e8e', f'{side}:plan_joint_roster_sha')
    if side == 'B':
        evidence['src/ireset/data/schema.py'] = {'status': 'unbound'}
    return {'revision': revision, 'files': evidence, 'joint_names': joint_names, 'site_names': site_names,
            'joint_qpos_addresses': list(range(7, 36)), 'joint_range_rad': ranges.tolist(),
            'joint_encoded_columns': [9, 38], 'site_order': 'KPT_TO_SMPLH insertion order',
            'axis_sentinels': 29, 'metrics_and_probe_ast_verified': True}
