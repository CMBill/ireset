"""Guard actual tensors by metadata, never cast, copy or execute a probe."""

import torch

from ireset.eval import w1_d6_joint_fit_protocol as p


FP32 = torch.float32
DISCRETE = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool)


def _facts(rows, device):
    device = torch.device(device)
    floats = total = 0
    for name, tensor, dtypes in rows:
        p.require(torch.is_tensor(tensor) and tensor.device == device and tensor.layout == torch.strided
                  and tensor.dtype in dtypes, f'real_tensor_contract:{name}:expected_{device}:{dtypes}')
        total += 1
        floats += int(tensor.dtype is FP32)
    p.require(floats > 0, 'real_tensor_contract_empty')
    return {'device': str(device), 'float_dtype': str(FP32), 'float_tensors': floats, 'tensors': total}


def module_facts(module, device):
    def rows():
        for name, tensor in module.named_parameters():
            yield 'parameter:' + name, tensor, (FP32,)
        for name, tensor in module.named_buffers():
            yield 'buffer:' + name, tensor, (FP32, *DISCRETE)

    return _facts(rows(), device)


def checkpoint_facts(checkpoint, device):
    def rows():
        for side in ('model', 'conditioner'):
            p.require(isinstance(checkpoint.get(side), dict) and bool(checkpoint[side]), 'real_checkpoint_tensor_roster:' + side)
            for name, tensor in checkpoint[side].items():
                yield f'{side}.{name}', tensor, (FP32, *DISCRETE)

    return _facts(rows(), device)


def check_state_destination(module, state, *, unit):
    # Both saved endpoints passed preflight. Also compare to this live
    # destination before each copy: strict=True alone would silently cast.
    live = unit('strict_destination_state', module.state_dict)
    def same_types():
        p.require(live.keys() == state.keys() and all(
            live[k].dtype == state[k].dtype and live[k].shape == state[k].shape for k in live),
            'real_strict_destination_state_mismatch')
    unit('strict_destination_types', same_types)


def fk_chain_facts(chain, device):
    # The separately pinned pytorch_kinematics Chain stores its actual FK
    # buffers as attributes rather than nn.Module buffers. Visual meshes and
    # parse-only metadata are not consumed by the tensor FK.
    p.require(torch.device(chain.device) == torch.device(device) and chain.dtype is FP32, 'real_fk_chain_declaration')
    def rows():
        for key in ('identity', 'low', 'high', 'axes', '_static_offsets', '_dof_ancestor_mask'):
            yield key, getattr(chain, key), (FP32,)
        for key in ('joint_type_indices', 'joint_indices', '_direct_parent_idx', '_joint_indices_clamped', '_dof_frame_indices'):
            yield key, getattr(chain, key), (torch.int64,)
        yield '_dof_is_revolute', chain._dof_is_revolute, (torch.bool,)
        for key in ('link_offsets', 'joint_offsets', 'parents_indices', '_bfs_levels'):
            for i, tensor in enumerate(getattr(chain, key)):
                if tensor is not None or key in ('parents_indices', '_bfs_levels'):
                    yield f'{key}[{i}]', tensor, (torch.int64,) if key in ('parents_indices', '_bfs_levels') else (FP32,)

    return _facts(rows(), device)


def loss_facts(loss, device):
    p.require(torch.device(loss.device) == torch.device(device), 'real_loss_device')
    return _facts(((key, getattr(loss, key), (FP32,)) for key in ('mean', 'std', 'lo', 'hi', 'R_g', 'x_g')), device)
