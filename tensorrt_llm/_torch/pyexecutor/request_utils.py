"""Utility functions for request processing."""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .scheduler import WaitingQueue

import torch

from tensorrt_llm._utils import maybe_pin_memory, nvtx_range
from tensorrt_llm.inputs.multimodal import _CPU_ONLY_MULTIMODAL_DATA_KEYS
from tensorrt_llm.mapping import CpType

from ..distributed import Distributed
from .hang_detector import HangDetector
from .llm_request import ExecutorRequest, LlmRequest, executor_request_to_llm_request

# Type alias for request queue items (to avoid circular import)
# The actual RequestQueueItem class is defined in executor_request_queue.py


def get_num_child_requests(request: ExecutorRequest) -> int:
    """Get the number of child requests for a given request.

    Args:
        request: The executor request to check.

    Returns:
        Number of child requests (0 if beam search, otherwise num_return_sequences - 1).
    """
    sampling_config = request.sampling_config
    return 0 if sampling_config.beam_width > 1 else (sampling_config.num_return_sequences or 1) - 1


def collect_py_objects_from_requests(
    requests: List, attribute_name: str
) -> Optional[Tuple[str, Dict]]:
    """Collect Python-only objects from requests.

    Args:
        requests: List of RequestQueueItem objects.
        attribute_name: Name of the attribute to collect.

    Returns:
        Tuple of (attribute_name, dict mapping request_id to object) or None if empty.
    """
    req_id_to_obj = {}
    for item in requests:
        if not item.is_normal_request:
            continue
        if item.request:
            obj = getattr(item.request, attribute_name, None)
            if obj is not None:
                req_id_to_obj[item.id] = obj
    return None if not req_id_to_obj else (attribute_name, req_id_to_obj)


def attach_py_objects_to_requests(requests: List, py_request_objects: Tuple) -> None:
    """Attach Python-only objects to each request.

    Args:
        requests: List of RequestQueueItem objects.
        py_request_objects: Tuple of (attribute_name, dict) pairs.
    """
    for attr_name, req_obj_dict in py_request_objects:
        for item in requests:
            if item.request:
                py_obj = req_obj_dict.get(item.id)
                if py_obj is not None:
                    setattr(item.request, attr_name, py_obj)


def derive_attention_dp_per_rank_request_cap(
    base_cap: int,
    max_num_tokens: Optional[int],
    max_total_draft_tokens: int,
) -> int:
    """Cap per-rank requests at ``max_num_tokens // (1 + max_total_draft_tokens)``
    so gen-phase per-step token load cannot exceed ``max_num_tokens`` under
    attention DP, where no component otherwise enforces a per-rank token cap
    (nvbug-6133201). Each gen request occupies ``1 + max_total_draft_tokens``
    token slots per step. Mirrors the CUDA graph batch-size cap at
    ``model_engine._filter_cuda_graph_batch_sizes``.

    Args:
        base_cap: Per-rank request cap from ``get_max_num_sequences()``.
        max_num_tokens: ``LlmArgs.max_num_tokens``; ``None`` disables tightening.
        max_total_draft_tokens: Draft tokens per gen request (0 without spec
            decoding); negative values are clamped to 0.

    Returns:
        The tighter of ``base_cap`` and ``max_num_tokens // step_tokens``.
    """
    if max_num_tokens is None:
        return base_cap
    step_tokens_per_req = 1 + max(max_total_draft_tokens, 0)
    return min(base_cap, max_num_tokens // step_tokens_per_req)


def can_process_attention_dp_request(
    req_item, all_ranks_num_active_requests: List[int], max_num_active_requests: int
) -> bool:
    """Check if a request can be processed immediately for attention DP.

    Args:
        req_item: The request queue item to check.
        all_ranks_num_active_requests: Number of active requests for each rank.
        max_num_active_requests: Maximum number of active requests per rank.

    Returns:
        True if the request can be processed, False otherwise.
    """
    scheduling_params = getattr(req_item.request, "py_scheduling_params", None)
    if scheduling_params is None:
        return True

    target_dp_rank = scheduling_params.attention_dp_rank
    if target_dp_rank is None or scheduling_params.attention_dp_relax:
        return True

    if all_ranks_num_active_requests[target_dp_rank] < max_num_active_requests:
        all_ranks_num_active_requests[target_dp_rank] += 1
        return True

    return False


def get_from_waiting_queue(
    waiting_queue: "WaitingQueue",
    max_req_count: int,
    enable_attention_dp: bool,
    max_num_active_requests: int,
    all_ranks_num_active_requests: Optional[List[int]] = None,
) -> List:
    """Get requests from the waiting queue.

    Args:
        waiting_queue: The queue to pop items from.
        max_req_count: Maximum items to retrieve. Returns empty list if <=0.
        enable_attention_dp: Whether to enable attention DP scheduling.
        max_num_active_requests: Maximum number of active requests per rank.
        all_ranks_num_active_requests: Number of active requests for each rank.

    Returns:
        List of requests that can be processed.
    """
    if max_req_count <= 0:
        return []

    req_count = 0
    items = []
    pending_requests = []

    # Track the request with strict requirements
    scheduling_all_ranks_num_active_requests = (
        all_ranks_num_active_requests.copy() if enable_attention_dp else None
    )

    while req_count < max_req_count and waiting_queue:
        req_item = waiting_queue.peek_request()
        num_children = len(req_item.child_req_ids) if req_item.child_req_ids else 0
        if (req_count + 1 + num_children) > max_req_count:
            break
        req_item = waiting_queue.pop_request()

        can_process = (
            can_process_attention_dp_request(
                req_item, scheduling_all_ranks_num_active_requests, max_num_active_requests
            )
            if enable_attention_dp
            else True
        )

        if can_process:
            items.append(req_item)
            req_count += 1 + num_children
        else:
            pending_requests.append(req_item)

    # Put the pending requests back to the waiting queue
    # All ranks should have the same waiting queue
    waiting_queue.prepend_requests(pending_requests)

    return items


def partition_context_for_star_attention(
    ctx_ids_list: List[int], cp_rank: int, cp_size: int, block_size: int, anchor_block_size: int
) -> Tuple[List[List[int]], List[List[int]], int]:
    """Partition context for Star Attention CP.

    Args:
        ctx_ids_list: List of context token IDs.
        cp_rank: Current CP rank.
        cp_size: Total number of CP ranks.
        block_size: Size of each block.
        anchor_block_size: Size of anchor block.

    Returns:
        Tuple of (ctx_blocks, position_blocks, padding).
    """
    ctx_ids = torch.tensor(ctx_ids_list).unsqueeze(0)
    ctx_len = ctx_ids.shape[-1]

    if block_size is None:
        block_size = ctx_len // cp_size
    if anchor_block_size is None:
        anchor_block_size = block_size

    assert anchor_block_size <= block_size, (
        f"cp_anchor_size {anchor_block_size} should be smaller than block_size {block_size}"
    )

    padding = 0
    if ctx_len % block_size != 0:
        padding = block_size - (ctx_len % block_size)
        assert padding <= ctx_len, "block size is too large for context, please set it smaller"
        ctx_ids = torch.cat((ctx_ids, torch.zeros_like(ctx_ids)[:, :padding]), dim=-1)
    position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0)

    ctx_ids_blocks = torch.tensor_split(torch.stack(ctx_ids.split(block_size, dim=-1)), cp_size)
    position_ids_blocks = torch.tensor_split(
        torch.stack(position_ids.split(block_size, dim=-1)), cp_size
    )

    if cp_rank != 0:
        ctx_blocks = [ctx_ids_blocks[0][0].tolist()[0][:anchor_block_size]]
        position_blocks = [position_ids_blocks[0][0].tolist()[0][:anchor_block_size]]
    else:
        ctx_blocks, position_blocks = [], []

    for idx in range(len(ctx_ids_blocks[cp_rank])):
        ctx_block = ctx_ids_blocks[cp_rank][idx]
        position_block = position_ids_blocks[cp_rank][idx]
        ctx_blocks.append(ctx_block.tolist()[0])
        position_blocks.append(position_block.tolist()[0])

    return ctx_blocks, position_blocks, padding


def partition_context_for_helix(
    input_token_ids: List[int], cp_rank: int, cp_size: int, tokens_per_block: int
) -> Tuple[List[int], List[int], int, int]:
    """Partition context for Helix CP.

    Args:
        input_token_ids: List of input token IDs.
        cp_rank: Current CP rank.
        cp_size: Total number of CP ranks.
        tokens_per_block: Number of tokens per block.

    Returns:
        Tuple of (input_ids_this_rank, position_ids_this_rank, input_len, padding_len).

    Raises:
        ValueError: If there aren't enough tokens for at least one block per CP rank.
    """
    all_input_ids = torch.tensor(input_token_ids, dtype=torch.int64).unsqueeze(0)
    input_len = all_input_ids.shape[-1]

    num_total_blocks = (input_len + tokens_per_block - 1) // tokens_per_block
    if num_total_blocks < cp_size:
        raise ValueError(
            f"There aren't enough tokens to get at least one block per CP rank. "
            f"num_total_blocks {num_total_blocks} < num_cp_ranks {cp_size}. "
            f"Please use smaller tokens_per_block for KV cache or reduce the number of CP ranks."
        )

    # Pad the last (partial) block so every block has exactly tokens_per_block tokens.
    padding_len = 0
    if input_len % tokens_per_block != 0:
        padding_len = tokens_per_block - (input_len % tokens_per_block)
        padding_ids = torch.zeros([1, padding_len], dtype=torch.int64)
        all_input_ids = torch.cat((all_input_ids, padding_ids), dim=-1)
    all_position_ids = torch.arange(0, input_len + padding_len, dtype=torch.int64).unsqueeze(0)

    # Round-robin block assignment across CP ranks: rank r owns blocks {r, r+cp_size, r+2*cp_size, ...}.
    # This must agree with the C++ KV cache split kernels (cacheSplitConcat.cu) so that the input
    # tokens this rank processes correspond to the KV blocks it received from the context server.
    input_id_blocks = list(all_input_ids.split(tokens_per_block, dim=-1))
    position_id_blocks = list(all_position_ids.split(tokens_per_block, dim=-1))

    input_ids_this_rank = torch.cat(input_id_blocks[cp_rank::cp_size], dim=-1).flatten().tolist()
    position_ids_this_rank = (
        torch.cat(position_id_blocks[cp_rank::cp_size], dim=-1).flatten().tolist()
    )

    # The (single) padded block is the global last block; under round-robin it is owned by rank
    # (num_total_blocks - 1) % cp_size, and is the last local block on that rank. Strip its padding.
    last_block_owner = (num_total_blocks - 1) % cp_size
    if cp_rank == last_block_owner and padding_len > 0:
        input_ids_this_rank = input_ids_this_rank[:-padding_len]
        position_ids_this_rank = position_ids_this_rank[:-padding_len]

    return input_ids_this_rank, position_ids_this_rank, input_len, padding_len


def merge_requests_to_llm_requests(
    new_requests: List, exclude_last_generation_logits: bool
) -> List[LlmRequest]:
    """Merge RequestQueueItems to LlmRequests (basic case without CP).

    Args:
        new_requests: List of RequestQueueItem objects.
        exclude_last_generation_logits: Whether to exclude last generation logits.

    Returns:
        List of LlmRequest objects including child requests.
    """
    req_with_children = []
    for req_item in new_requests:
        req = executor_request_to_llm_request(
            req_item.id, req_item.request, req_item.child_req_ids, exclude_last_generation_logits
        )
        req_with_children.append(req)
        if req.child_requests:
            req_with_children.extend(req.child_requests)
    return req_with_children


def merge_helix_requests(
    new_requests: List,
    cp_rank: int,
    cp_size: int,
    tokens_per_block: int,
    exclude_last_generation_logits: bool,
) -> List[LlmRequest]:
    """Merge requests for Helix CP.

    Note: Helix parallelism is a decode-only feature run with disaggregated serving.
    This function gets called on gen server during initialization of a new request.

    Args:
        new_requests: List of RequestQueueItem objects.
        cp_rank: Current CP rank.
        cp_size: Total number of CP ranks.
        tokens_per_block: Number of tokens per block.
        exclude_last_generation_logits: Whether to exclude last generation logits.

    Returns:
        List of LlmRequest objects including child requests.
    """
    req_with_children = []

    for req_item in new_requests:
        input_ids_this_rank, position_ids_this_rank, input_len, _ = partition_context_for_helix(
            req_item.request.input_token_ids, cp_rank, cp_size, tokens_per_block
        )

        req = executor_request_to_llm_request(
            req_id=req_item.id,
            executor_request=req_item.request,
            child_req_ids=req_item.child_req_ids,
            exclude_last_generation_logits=exclude_last_generation_logits,
            input_token_ids=input_ids_this_rank,
            position_ids=position_ids_this_rank,
        )
        req.total_input_len_cp = input_len
        req.seqlen_this_rank_cp = len(input_ids_this_rank)
        req_with_children.append(req)
        if req.child_requests:
            req_with_children.extend(req.child_requests)

    return req_with_children


def merge_star_attention_requests(
    new_requests: List,
    cp_rank: int,
    cp_size: int,
    cp_config: dict,
    exclude_last_generation_logits: bool,
) -> List[LlmRequest]:
    """Merge requests for Star Attention CP.

    Args:
        new_requests: List of RequestQueueItem objects.
        cp_rank: Current CP rank.
        cp_size: Total number of CP ranks.
        cp_config: CP configuration dict containing 'block_size' and 'cp_anchor_size'.
        exclude_last_generation_logits: Whether to exclude last generation logits.

    Returns:
        List of LlmRequest objects.
    """
    result = []
    block_size = cp_config["block_size"]
    anchor_block_size = cp_config["cp_anchor_size"]

    for req_item in new_requests:
        req_id, exe_req, query_token_ids = req_item.id, req_item.request, req_item.query
        ctx_len0 = len(exe_req.input_token_ids)

        ctx_blocks, position_blocks, last_block_padding_num = partition_context_for_star_attention(
            exe_req.input_token_ids, cp_rank, cp_size, block_size, anchor_block_size
        )

        if cp_rank == cp_size - 1 and last_block_padding_num > 0:
            ctx_blocks[-1] = ctx_blocks[-1][:-last_block_padding_num]
            position_blocks[-1] = position_blocks[-1][:-last_block_padding_num]

        # if has query
        if query_token_ids:
            ctx_blocks.append(query_token_ids)
            position_blocks.append([i for i in range(ctx_len0, ctx_len0 + len(query_token_ids))])

        # insert the dummy block to align the number of ctx iterations of each rank
        total_blocks = (ctx_len0 + block_size - 1) // block_size
        num_blocks_per_rank = (total_blocks + cp_size - 1) // cp_size + 1  # 1 for query block
        if len(ctx_blocks) == num_blocks_per_rank:
            ctx_blocks.insert(1, [])
            position_blocks.insert(1, [])
        elif len(ctx_blocks) == num_blocks_per_rank + 1:
            # anchor + ctx_blocks + qry_block
            pass
        else:
            raise ValueError(
                f"Invalid context partition: rank = {cp_rank}, "
                f"len(ctx_blocks) = {len(ctx_blocks)}, "
                f"num_blocks_per_rank = {num_blocks_per_rank}"
            )

        # fake data for scheduler
        ctx_blocks_list = [0] * (block_size + anchor_block_size)

        req = executor_request_to_llm_request(
            req_id, exe_req, exclude_last_generation_logits, ctx_blocks_list
        )
        req.gen_iters = 0
        req.ctx_iters = 0
        req.ctx_blocks = ctx_blocks
        req.ctx_position_blocks = position_blocks
        req.query_id = query_token_ids

        result.append(req)

    return result


@nvtx_range("merge_requests")
def merge_requests(
    new_requests: List,
    cp_config: dict,
    cp_rank: int,
    cp_size: int,
    exclude_last_generation_logits: bool,
) -> List[LlmRequest]:
    """Merge RequestQueueItems to LlmRequests based on CP configuration.

    This is a router function that dispatches to the appropriate merge function
    based on the CP (Context Parallelism) configuration.

    Args:
        new_requests: List of RequestQueueItem objects.
        cp_config: CP configuration dict. May contain 'cp_type', 'tokens_per_block',
            'block_size', 'cp_anchor_size'.
        cp_rank: Current CP rank.
        cp_size: Total number of CP ranks.
        exclude_last_generation_logits: Whether to exclude last generation logits.

    Returns:
        List of LlmRequest objects.

    Raises:
        NotImplementedError: If cp_type is not supported.
    """
    if "cp_type" in cp_config:
        cp_type = cp_config["cp_type"]
        if cp_type == CpType.STAR:
            return merge_star_attention_requests(
                new_requests,
                cp_rank=cp_rank,
                cp_size=cp_size,
                cp_config=cp_config,
                exclude_last_generation_logits=exclude_last_generation_logits,
            )
        elif cp_type == CpType.HELIX:
            return merge_helix_requests(
                new_requests,
                cp_rank=cp_rank,
                cp_size=cp_size,
                tokens_per_block=cp_config["tokens_per_block"],
                exclude_last_generation_logits=exclude_last_generation_logits,
            )
        else:
            raise NotImplementedError(f"Unsupported cp type {cp_type.name}.")

    return merge_requests_to_llm_requests(new_requests, exclude_last_generation_logits)


_NCCL_MM_BCAST_MIN_BYTES = 64 * 1024
_NCCL_MM_BCAST_ALIGNMENT = 256


@dataclass(frozen=True, eq=False)
class _DeviceTensorRef:
    nbytes: int
    dtype: str
    shape: Tuple[int, ...]
    source: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)

    def __reduce__(self):
        # The source stays attached to rank 0's queued request. Other ranks
        # receive only the descriptor and allocate the destination at admission.
        return (_rebuild_device_tensor_ref, (self.nbytes, self.dtype, self.shape))


def _rebuild_device_tensor_ref(nbytes: int, dtype: str, shape: Tuple[int, ...]) -> _DeviceTensorRef:
    return _DeviceTensorRef(nbytes=nbytes, dtype=dtype, shape=shape)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class _MultimodalTensorPacker:
    """Replace GPU-bound multimodal tensors with flat-buffer references."""

    def __init__(self, device_paths: Optional[Sequence[str]]):
        self._device_paths = (
            None if device_paths is None else tuple(tuple(path.split(".")) for path in device_paths)
        )
        self._refs = {}
        self.packed = False

    def pack(self, value: Any) -> Any:
        return self._pack(value, ())

    def _pack(self, value: Any, path: Tuple[Any, ...]) -> Any:
        if type(value) is torch.Tensor:
            if self._should_pack(value, path):
                return self._add(value)
            return value
        if isinstance(value, dict):
            return {key: self._pack(child, path + (key,)) for key, child in value.items()}
        if isinstance(value, list):
            return [self._pack(child, path) for child in value]
        return value

    def _should_pack(self, tensor: torch.Tensor, path: Tuple[Any, ...]) -> bool:
        if (
            tensor.device.type not in ("cpu", "cuda")
            or tensor.layout != torch.strided
            or tensor.is_quantized
            or tensor.is_nested
            or tensor.numel() == 0
        ):
            return False

        if tensor.device.type == "cuda":
            return True

        # Precomputed embeddings are GPU inputs even when a model leaves its
        # generic device paths empty so the fallback path can slice first.
        if path == ("multimodal_embedding",):
            return True
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes < _NCCL_MM_BCAST_MIN_BYTES:
            return False
        if self._device_paths is None:
            return not any(key in _CPU_ONLY_MULTIMODAL_DATA_KEYS for key in path)
        return any(path[: len(target)] == target for target in self._device_paths)

    def _add(self, tensor: torch.Tensor) -> _DeviceTensorRef:
        key = id(tensor)
        if key in self._refs:
            return self._refs[key]

        ref = _DeviceTensorRef(
            nbytes=tensor.numel() * tensor.element_size(),
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(tensor.shape),
            source=tensor,
        )
        self._refs[key] = ref
        self.packed = True
        return ref


def _device_tensor_view(flat: torch.Tensor, ref: _DeviceTensorRef, offset: int) -> torch.Tensor:
    dtype = getattr(torch, ref.dtype)
    return flat.narrow(0, offset, ref.nbytes).view(dtype).view(ref.shape)


def _layout_device_tensors(
    value: Any,
) -> Tuple[List[Tuple[_DeviceTensorRef, int]], int]:
    placements = []
    seen = set()
    total_bytes = 0

    def collect(child: Any) -> None:
        nonlocal total_bytes
        if isinstance(child, _DeviceTensorRef):
            if id(child) not in seen:
                seen.add(id(child))
                offset = _align_up(total_bytes, _NCCL_MM_BCAST_ALIGNMENT)
                placements.append((child, offset))
                total_bytes = offset + child.nbytes
        elif isinstance(child, dict):
            for nested in child.values():
                collect(nested)
        elif isinstance(child, list):
            for nested in child:
                collect(nested)

    collect(value)
    return placements, total_bytes


def _restore_device_tensors(value: Any, views: Dict[int, torch.Tensor]) -> Any:
    if isinstance(value, _DeviceTensorRef):
        return views[id(value)]
    if isinstance(value, dict):
        return {key: _restore_device_tensors(child, views) for key, child in value.items()}
    if isinstance(value, list):
        return [_restore_device_tensors(child, views) for child in value]
    return value


def _pack_multimodal_tensors(
    py_request_objects: Optional[Tuple], device_paths: Optional[Sequence[str]]
) -> Optional[Tuple]:
    packer = _MultimodalTensorPacker(device_paths)
    if not py_request_objects:
        return py_request_objects

    packed_objects = []
    for attr_name, request_objects in py_request_objects:
        if attr_name == "py_multimodal_data":
            request_objects = {
                request_id: packer.pack(multimodal_data)
                for request_id, multimodal_data in request_objects.items()
            }
        packed_objects.append((attr_name, request_objects))
    return tuple(packed_objects) if packer.packed else py_request_objects


class RequestBroadcaster:
    """Broadcast requests and their Python-only payloads across ranks.

    Large GPU-bound multimodal tensors are sent as MPI descriptors while
    queued, then materialized into one NCCL-broadcast device buffer when the
    requests are admitted. Unsupported configurations retain the MPI path.
    """

    def __init__(
        self,
        dist: Distributed,
        hang_detector: HangDetector,
        execution_stream: Optional[torch.cuda.Stream] = None,
        multimodal_data_device_paths: Optional[Sequence[str]] = None,
        enable_attention_dp: bool = False,
    ):
        self.dist = dist
        self.hang_detector = hang_detector
        self.execution_stream = execution_stream
        self.multimodal_data_device_paths = multimodal_data_device_paths
        self.enable_attention_dp = enable_attention_dp
        self.send_requests_handler = None
        self._nccl_broadcast_enabled: Optional[bool] = None
        self._nccl_broadcast_overlap_enabled = False
        self._nccl_broadcast_stream = None

    def broadcast(self, new_requests: List) -> Tuple[List, Optional[Tuple]]:
        """Return broadcast requests and their Python payloads."""
        if self.dist.rank == 0:
            py_request_objects = self._collect_py_objects(new_requests)
        else:
            py_request_objects = None

        if self.dist.world_size == 1:
            return new_requests, py_request_objects

        with self.hang_detector.pause():
            nccl_broadcast = self._use_nccl_broadcast()
        if nccl_broadcast:
            with self.hang_detector.pause():
                new_requests, py_request_objects = self._broadcast_tensor_descriptors(
                    new_requests, py_request_objects
                )
            return new_requests, py_request_objects

        if self.dist.rank == 0:
            self._broadcast_requests(new_requests, py_request_objects)
        else:
            with self.hang_detector.pause():
                new_requests, py_request_objects = self._broadcast_requests(
                    new_requests, py_request_objects
                )
        return new_requests, py_request_objects

    def drain(self) -> None:
        """Complete any in-flight device broadcast."""
        if self._nccl_broadcast_stream is not None:
            self._nccl_broadcast_stream.synchronize()

    def materialize_multimodal_tensors(self, new_requests: List) -> None:
        """Materialize admitted multimodal tensors on every rank's GPU."""
        if not self._nccl_broadcast_enabled:
            return

        requests = [
            item.request
            for item in new_requests
            if item.is_normal_request
            and item.request is not None
            and getattr(item.request, "py_multimodal_data", None) is not None
        ]
        placements, total_bytes = _layout_device_tensors(
            [request.py_multimodal_data for request in requests]
        )
        if not placements:
            return

        flat = self._broadcast_device_tensors(total_bytes, placements)
        views = {id(ref): _device_tensor_view(flat, ref, offset) for ref, offset in placements}
        for request in requests:
            request.py_multimodal_data = _restore_device_tensors(request.py_multimodal_data, views)

    def _use_nccl_broadcast(self) -> bool:
        if self._nccl_broadcast_enabled is None:
            local = (
                hasattr(self.dist, "broadcast_device")
                and self.execution_stream is not None
                and torch.cuda.is_available()
                and not self.dist.has_pp
                and not self.enable_attention_dp
                and os.environ.get("TLLM_DISABLE_NCCL_MM_BCAST", "0") != "1"
            )
            implicit_launch_order = os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT", "0") == "1"
            rank_settings = self.dist.allgather((local, implicit_launch_order))
            # All ranks must use the same collective sequence. Overlap also
            # requires implicit launch ordering to be enabled on every rank.
            self._nccl_broadcast_enabled = all(enabled for enabled, _ in rank_settings)
            self._nccl_broadcast_overlap_enabled = self._nccl_broadcast_enabled and all(
                overlap for _, overlap in rank_settings
            )
        return self._nccl_broadcast_enabled

    def _order_nccl_broadcast_stream(
        self,
        stream: torch.cuda.Stream,
        current_stream: torch.cuda.Stream,
        placements: Sequence[Tuple[_DeviceTensorRef, int]],
    ) -> None:
        if not self._nccl_broadcast_overlap_enabled:
            stream.wait_stream(self.execution_stream)
            stream.wait_stream(current_stream)
        elif any(
            ref.source is not None and ref.source.device.type == "cuda" for ref, _ in placements
        ):
            stream.wait_stream(current_stream)

    def _broadcast_tensor_descriptors(
        self, new_requests: List, py_request_objects: Optional[Tuple]
    ) -> Tuple[List, Optional[Tuple]]:
        if self.dist.rank == 0:
            py_request_objects = _pack_multimodal_tensors(
                py_request_objects, self.multimodal_data_device_paths
            )

        payload = (new_requests, py_request_objects)
        with nvtx_range("broadcast_requests"):
            payload = self.dist.broadcast(payload, root=0)
        return payload

    def _broadcast_device_tensors(
        self,
        total_bytes: int,
        placements: Sequence[Tuple[_DeviceTensorRef, int]],
    ) -> torch.Tensor:
        if self._nccl_broadcast_stream is None:
            self._nccl_broadcast_stream = torch.cuda.Stream()

        stream = self._nccl_broadcast_stream
        current_stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            # Allocate on the first-use stream so a caching-allocator reuse is
            # ordered before the H2D copies queued below.
            flat = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
            for ref, offset in placements:
                if ref.source is not None and ref.source.device.type == "cpu":
                    _device_tensor_view(flat, ref, offset).copy_(
                        maybe_pin_memory(ref.source), non_blocking=True
                    )

            # NCCL implicit launch ordering permits the dedicated communicator
            # to overlap the previous forward without cross-communicator
            # deadlocks. CUDA sources still wait for their producing stream.
            self._order_nccl_broadcast_stream(stream, current_stream, placements)
            for ref, offset in placements:
                if ref.source is not None and ref.source.device.type == "cuda":
                    _device_tensor_view(flat, ref, offset).copy_(ref.source, non_blocking=True)
                    ref.source.record_stream(stream)

            self.dist.broadcast_device(flat, root=0)
            complete = torch.cuda.Event()
            complete.record()

        self.execution_stream.wait_event(complete)
        current_stream.wait_event(complete)
        flat.record_stream(self.execution_stream)
        flat.record_stream(current_stream)
        return flat

    def _collect_py_objects(self, new_requests: List) -> Tuple:
        """Collect Python-only objects from requests."""
        py_logits_post_processors = collect_py_objects_from_requests(
            new_requests, "py_logits_post_processors"
        )
        py_multimodal_data = collect_py_objects_from_requests(new_requests, "py_multimodal_data")
        py_scheduling_params = collect_py_objects_from_requests(
            new_requests, "py_scheduling_params"
        )
        py_num_logprobs = collect_py_objects_from_requests(new_requests, "py_num_logprobs")
        py_dynamic_temperature_rules = collect_py_objects_from_requests(
            new_requests, "py_dynamic_temperature_rules"
        )
        py_disaggregated_params = collect_py_objects_from_requests(
            new_requests, "py_disaggregated_params"
        )
        py_lora_path = collect_py_objects_from_requests(new_requests, "py_lora_path")
        py_external_request_id = collect_py_objects_from_requests(
            new_requests, "py_external_request_id"
        )

        return tuple(
            filter(
                None,
                [
                    py_logits_post_processors,
                    py_multimodal_data,
                    py_scheduling_params,
                    py_num_logprobs,
                    py_dynamic_temperature_rules,
                    py_disaggregated_params,
                    py_lora_path,
                    py_external_request_id,
                ],
            )
        )

    @nvtx_range("broadcast_requests")
    def _broadcast_requests(
        self, new_requests: List, py_request_objects
    ) -> Tuple[List, Optional[Dict]]:
        """Broadcast requests across pipeline stages."""
        payloads = (new_requests, py_request_objects)

        if self.dist.world_size == 1:
            return payloads

        if not self.dist.has_pp:
            return self.dist.broadcast(payloads, root=0)

        # Broadcast within first PP stage before send/recv chain to other PP stages.
        # This needs to cover both TP and CP ranks within the first PP stage.
        if self.dist.is_first_pp_rank:
            with nvtx_range("tp_broadcast_requests"):
                payloads = self.dist.tp_cp_broadcast(payloads, root=0)

        # Tag for communication
        tag = self.dist.pp_size  # Use pp_size as tag to avoid conflicts

        # Send payloads
        if not self.dist.is_first_pp_rank:
            with nvtx_range("recv_requests_from_prev_pp"):
                payloads = self.dist.recv_object(self.dist.prev_pp_rank, tag)

        # isend new requests may cause deadlock, when CUDA_LAUNCH_BLOCKING=1
        # or PP microbatches can't overlap, the deadlock will happen:
        # 1. rank1 will wait on nccl.send(rank2), without invoking mpi.wait(isend-handle)
        # 2. rank2 will wait on mpi.recv(rank1) but never receive the new requests.
        # 3. rank1 will hang on nccl.send because rank2 will never reach nccl.recv(rank1).
        pp_send_func = (
            self.dist.isend_object
            if os.environ.get("TRTLLM_PP_REQ_SEND_ASYNC", "0") == "1"
            else self.dist.send_object
        )

        if not self.dist.is_last_pp_rank:
            if self.send_requests_handler is not None:
                with nvtx_range("wait_prev_send_requests_handler"):
                    self.send_requests_handler.wait()
            with nvtx_range("send_requests_to_next_pp"):
                self.send_requests_handler = pp_send_func(payloads, self.dist.next_pp_rank, tag)

        return payloads
