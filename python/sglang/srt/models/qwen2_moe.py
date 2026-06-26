# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""

import logging
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.distributed import (
    get_moe_data_parallel_world_size,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.parallel_state import (
    get_attn_context_model_parallel_world_size,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK, TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_prefill_context_parallel_enabled,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    make_layers,
    use_intel_amx_backend,
)

if is_npu():
    from sglang.srt.hardware_backend.npu.cmo import (
        shared_expert_on_independent_stream,
        wait_share_stream,
    )

from sglang.srt.environ import envs
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


def can_fuse_shared_expert(
    config: PretrainedConfig,
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Whether the shared expert may be fused as an extra MoE expert (Qwen3.5 + Aiter).

    Caller must still gate on ``support_shared_expert_fusion`` and ``_use_aiter``.
    """
    if (
        get_global_server_args().disable_shared_experts_fusion is True
        or getattr(config, "shared_expert_intermediate_size", 0) <= 0
        or config.shared_expert_intermediate_size != config.moe_intermediate_size
        or get_moe_a2a_backend().is_deepep()
    ):
        return False

    # If the shared expert is excluded from quantization (stored as FP32 in the
    # checkpoint), fusing it into the quantized MoE weight tensor requires online
    # quantization which is not supported. Disable fusion in this case.
    if quant_config is not None:
        exclude_layers = getattr(quant_config, "exclude_layers", [])
        if any(
            "shared_expert" in layer
            and "shared_expert_gate" not in layer
            and not layer.startswith("mtp.")
            for layer in exclude_layers
        ):
            return False

    return True


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x

from collections import OrderedDict
import json
import os

EXP_DIR = os.getenv("EXP_DIR", os.path.join(os.path.abspath(os.path.curdir), "eval/experiments/tmp"))
with open(os.path.join(EXP_DIR, "exp_args.json"), "r") as f:
    exp_args = json.load(f)

class CacheRegistry:
    _reg = {}

    @staticmethod
    def add(cache):
        layer_id = cache.layer_id
        rid = cache.rid
        if rid not in CacheRegistry._reg: CacheRegistry._reg[rid] = {}
        assert layer_id not in CacheRegistry._reg[rid], f"cache with rid={rid} and layer_id={layer_id} already exists in registry."
        CacheRegistry._reg[rid][layer_id] = cache

    @staticmethod
    def get(rid, layer_id):
        assert rid in CacheRegistry._reg, f"rid={rid} not in registry."
        assert layer_id in CacheRegistry._reg[rid], f"rid={rid}, layer_id={layer_id} not in registry."
        return CacheRegistry._reg[rid][layer_id]

class Cache:
    def __init__(self, layer_id, rid, num_experts, static_cap, dynamic_cap, record_activations=False) -> None:
        self.static_cap = static_cap
        self.static_dat = OrderedDict()
        self.dynamic_cap = dynamic_cap
        self.dynamic_dat = OrderedDict()

        self.prefetched = []

        self.n_hits = 0
        self.n_miss = 0
        self.n_corr_pref = 0
        self.n_pref = 0
        self.layer_id = layer_id
        self.rid = rid
        self.num_experts = num_experts
        self.static_stats = {e: 0 for e in range(self.num_experts)}
        self.record_activations = record_activations
        self.activations = []

        self.hot_stats = {e: 0 for e in range(num_experts)}

        # Dynamic-cache eviction policy: "lru" (least-recently-used, default) or
        # "lfu" (least-frequently-used over the sequence; ties -> LRU). access_ct
        # tracks per-expert usage frequency for this request, used by LFU evict.
        self.policy = exp_args.get("cache_policy", "lru")
        self.access_ct = {}

        self.init_random()

    def __exit__(self, exc_type, exc, tb):
        print(f"Cache {self.layer_id} for {self.rid} exited")

    def init_random(self):
        # Static cache is seeded from per-layer hot-expert stats. Skip the file
        # read entirely when there's no static cache (static_cap=0, e.g. the
        # draft's own pure-dynamic cache) or when this layer has no stats (e.g.
        # the MTP draft layer is absent from hot_experts.json).
        if self.static_cap > 0:
            with open(exp_args["hot_experts_file"], "r") as f:
                _hot_experts = json.load(f).get(str(self.layer_id))
            if _hot_experts is not None:
                hot_experts = sorted(list(range(self.num_experts)), key=lambda e: _hot_experts[str(e)], reverse=True)
                for e in hot_experts[:self.static_cap]:
                    self.static_dat[e] = 0

        for e in range(self.dynamic_cap):
            self.dynamic_dat[e] = 0

    def evict(self):
        if len(self.dynamic_dat) < self.dynamic_cap:
            return
        if self.policy == "lfu":
            # Evict the least-frequently-used cached expert. dynamic_dat iterates
            # oldest-first, so min() returns the LRU among the lowest-frequency
            # experts -> frequency primary, recency tiebreak.
            victim = min(self.dynamic_dat, key=lambda e: self.access_ct.get(e, 0))
            del self.dynamic_dat[victim]
        else:  # "lru"
            self.dynamic_dat.popitem(last=False)

    def prefetch(self, prefetch_experts):
        self.prefetched = prefetch_experts

    def read(self, experts, is_decode):
        hits = [e for e in experts if e in self.static_dat or e in self.dynamic_dat or e in self.prefetched]
        misses = [e for e in experts if e not in hits]

        if is_decode:
            for e in experts: self.hot_stats[e] += 1

            self.n_hits += len(hits)
            self.n_miss += len(misses)
            self.n_pref += len(self.prefetched)
            self.n_corr_pref += len([e for e in self.prefetched if e in hits])
            if self.record_activations:
                self.activations.append(
                    {
                        "active_experts": experts,
                        "in_cache": list(self.static_dat.keys()) + list(self.dynamic_dat.keys()),
                        "prefetched": self.prefetched
                    }
                )

        if self.dynamic_cap > 0:
            self._update_lru(experts)

        return hits

    def _update_lru(self, experts):
        for e in experts:
            self.access_ct[e] = self.access_ct.get(e, 0) + 1   # per-sequence frequency (for LFU)
            if e in self.static_dat:
                pass
            elif e in self.dynamic_dat:
                self.dynamic_dat.move_to_end(e)   # recency (LRU + LFU tiebreak)
            else:
                self.evict()
                self.dynamic_dat[e] = 0

    def read_verify(self, tokens_experts):
        """Record one speculative *verify* step (K+1 candidate tokens).

        All candidate tokens route against the SAME frozen cache snapshot, then
        the dynamic LRU is updated with ALL experts used in verification (not
        just accepted ones) — decided to be fine and potentially better for
        future-activation prediction. `tokens_experts` is a list of per-token
        expert lists; recorded as a list-of-lists so parse_spec_decode can
        recover per-expert row counts. accept_len is captured separately from
        the API meta_info, not here (it is only known after the forward)."""
        resident = set(self.static_dat) | set(self.dynamic_dat) | set(self.prefetched)
        flat = [e for tex in tokens_experts for e in tex]
        hits = [e for e in flat if e in resident]

        for e in flat:
            self.hot_stats[e] += 1
        self.n_hits += len(hits)
        self.n_miss += len(flat) - len(hits)
        # prefetch precision is counted per candidate token, mirroring decode.
        self.n_pref += len(self.prefetched) * len(tokens_experts)
        self.n_corr_pref += len([e for e in flat if e in self.prefetched])

        if self.record_activations:
            self.activations.append(
                {
                    "active_experts": [list(tex) for tex in tokens_experts],
                    "in_cache": list(self.static_dat.keys()) + list(self.dynamic_dat.keys()),
                    "prefetched": self.prefetched,
                }
            )

        if self.dynamic_cap > 0:
            self._update_lru(flat)

    def get_experts_in_cache(self):
        return list(set(list(self.static_dat.keys()) + list(self.dynamic_dat.keys()) + self.prefetched))

    def dump_hot_stats(self):
        with open(os.path.join(EXP_DIR, "hot_stats.jsonl"), "a") as f:
            f.write(json.dumps(
                {
                    "layer_id": self.layer_id,
                    "rid": self.rid,
                    "hot_stats": self.hot_stats
                }
            ))
            f.write("\n")

    def dump_cache_stats(self):
        with open(os.path.join(EXP_DIR, "cache_stats.jsonl"), "a") as f:
            total = self.n_hits + self.n_miss
            hit_ratio = self.n_hits / total if total > 0 else 0
            f.write(json.dumps(
                {
                    "layer_id": self.layer_id,
                    "rid": self.rid,
                    "n_hits": self.n_hits,
                    "n_miss": self.n_miss,
                    "n_corr_pref": self.n_corr_pref,
                    "n_pref": self.n_pref,
                    "hit_ratio": hit_ratio
                }
            ))
            f.write("\n")

    def dump_activations(self):
        with open(os.path.join(EXP_DIR, "active_experts.jsonl"), "a") as f:
            f.write(json.dumps(
                {
                    "layer_id": self.layer_id,
                    "rid": self.rid,
                    "data": self.activations
                }
            ))
            f.write("\n")

    def flush(self):
        self.dump_cache_stats()
        self.dump_hot_stats()
        if self.record_activations:
            self.dump_activations()

class CacheAwareTopk(torch.nn.Module):
    def __init__(self, top_k, renormalize, layer_id, fixed_num_experts) -> None:
        super().__init__()
        assert renormalize == True, "renormalize == False is not yet supported"
        self.topk = top_k
        self.renormalize = renormalize
        self.layer_id = layer_id
        self.fixed_num_experts = fixed_num_experts

    def forward(self, hidden_states, router_logits, cached_experts=None):
        if cached_experts is None:
            # topk_ids = torch.topk(router_logits.float(), self.topk, dim=-1).indices.contiguous()

            # topk_weights = torch.gather(router_logits.float(), 1, topk_ids).softmax(dim=-1).contiguous()
            # assert topk_weights.is_contiguous()

            logits = router_logits.float()                          # upcast bf16/fp16 -> fp32, as the kernel does
            probs = torch.softmax(logits, dim=-1)                   # softmax over ALL experts, fp32

            order = torch.argsort(probs, dim=-1, descending=True, stable=True)
            topk_ids = order[:, :self.topk].to(torch.int32).contiguous()

            topk_weights = torch.gather(probs, 1, topk_ids.long())

            if self.renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            out = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
            )
        else:
            logits = router_logits.float()
            n_tokens, num_experts = logits.shape

            probs = torch.softmax(logits, dim=-1)

            topks = torch.argsort(probs, descending=True, dim=-1).to(torch.int32).contiguous()

            ar = torch.arange(num_experts, device=topks.device).unsqueeze(0)

            # cache_mask = torch.nn.functional.one_hot(torch.tensor(cached_experts, device=topks.device) , num_classes=num_experts).sum(dim=1).bool()
            # cache_mask[i, e] == True iff expert e is cached/prefetched for token i  (ragged-safe)
            lengths  = torch.tensor([len(c) for c in cached_experts], device=topks.device)
            col_idx  = torch.tensor([e for c in cached_experts for e in c],
                                    dtype=torch.long, device=topks.device)
            row_idx  = torch.repeat_interleave(
                torch.arange(n_tokens, device=topks.device), lengths)
            cache_mask = torch.zeros(n_tokens, num_experts, dtype=torch.bool, device=topks.device)
            cache_mask[row_idx, col_idx] = True

            # is the expert at sorted-position j cached for this token?
            cached_at_pos = torch.gather(cache_mask, 1, topks).contiguous()              # [n_tokens, K] bool

            # keep: first FIXED_EXPERTS always, plus cached ones among the rest
            keep = (ar < self.fixed_num_experts) | cached_at_pos                    # [n_tokens, K]

            # stable partition: kept positions keep their order up front, rest pushed back
            sort_key = torch.where(keep, ar, ar + num_experts)
            order = torch.argsort(sort_key, dim=1, stable=True)            # [n_tokens, K]

            selected = torch.gather(topks, 1, order)[:, :self.topk].contiguous()        # [n_tokens, topk]

            topk_weights = torch.gather(probs, dim=1, index=selected).contiguous()

            if self.renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            out = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=selected,
                router_logits=router_logits,
            )

        return out


RIDS = []




class EarlyGate:
    def __init__(self):
        self.gates = {}
        self.preds = {}
        self.prefetch_offset = exp_args["prefetch_offset"]
        self.n_preds = exp_args["n_prefetch"]
        self.topk = CacheAwareTopk(
            top_k=self.n_preds,
            renormalize=True,
            layer_id=0,
            fixed_num_experts=0
        )
        self.is_enabled = self.n_preds > 0

        # "early_gate" (default): run the future layer's gate on the current
        # hidden states. "trained": use the per-layer predictors trained in
        # the moe-activation-predictor repo (see expert_predictor.py).
        self.predictor_mode = exp_args.get("predictor", "early_gate")
        self.predictor_ckpt_root = exp_args.get("predictor_ckpt_root")
        self.predictors = {}        # target_layer_id -> model | None (None: no ckpt)
        if self.is_enabled and self.predictor_mode == "trained":
            assert self.predictor_ckpt_root is not None, (
                "exp_args['predictor_ckpt_root'] must be set when predictor='trained'"
            )
            self._validate_ckpt_root()

    def _validate_ckpt_root(self):
        # Fail loudly on a wrong/empty --predictor-ckpt-root. A bad path used to
        # make every load_predictor() return None, silently turning
        # predictor="trained" into "no prefetch" (indistinguishable from
        # cache-only). NOTE: early_gate has no path to validate — it runs the
        # model's own future-layer gate (add_gate), so only "trained" needs this.
        from sglang.srt.models.expert_predictor import predictor_dir

        root = self.predictor_ckpt_root
        if not os.path.isdir(root):
            raise FileNotFoundError(f"predictor_ckpt_root does not exist: {root!r}")

        # At least one target layer must have a checkpoint for the configured
        # offset; otherwise the path or offset is wrong. (A few legitimately
        # untrained tail layers staying empty is fine — that's handled, and
        # warned about, per-layer in _get_predictor.)
        for d in os.listdir(root):
            if not d.startswith("layer"):
                continue
            try:
                target = int(d[len("layer"):])
            except ValueError:
                continue
            ckpt = os.path.join(
                predictor_dir(root, target, self.prefetch_offset), "best.pt"
            )
            if os.path.exists(ckpt):
                return
        raise FileNotFoundError(
            f"No trained predictor checkpoints found under {root!r} for "
            f"prefetch_offset={self.prefetch_offset} (looked for "
            f"layer*/layer*_moe_hidden_states/latest/best.pt). Wrong path or offset?"
        )

    def _get_predictor(self, target_layer_id):
        # Lazily load (and cache) the predictor for target_layer_id. The root is
        # validated up front (_validate_ckpt_root), so a None here means this
        # specific (layer, offset) genuinely has no checkpoint (e.g. the
        # untrained tail layers) -> warn once, no prefetch for that layer.
        if target_layer_id not in self.predictors:
            from sglang.srt.models.expert_predictor import load_predictor

            pred = load_predictor(
                self.predictor_ckpt_root,
                target_layer_id,
                self.prefetch_offset,
                device=torch.device("cuda"),
            )
            if pred is None:
                logger.warning(
                    "No trained predictor for target_layer=%d (offset=%d); "
                    "no prefetch for this layer.",
                    target_layer_id,
                    self.prefetch_offset,
                )
            self.predictors[target_layer_id] = pred
        return self.predictors[target_layer_id]

    def add_gate(self, layer_id, gate):
        if not self.is_enabled: return
        assert layer_id not in self.gates
        self.gates[layer_id] = gate

    def _get_not_cached_experts(self, target_layer_id, rids, num_experts):
        assert rids is not None, "rids must be provided for cache-aware prefetch"
        cached_experts = [
            CacheRegistry.get(rid, target_layer_id).get_experts_in_cache()
            for rid in rids
        ]
        return [
            [e for e in range(num_experts) if e not in cached_experts[i]]
            for i in range(len(cached_experts))
        ]

    def make_pred(self, layer_id, hidden_states, rids=None):
        if not self.is_enabled: return

        next_layer_id = layer_id+self.prefetch_offset

        if self.predictor_mode == "trained":
            predictor = self._get_predictor(next_layer_id)
            if predictor is not None:
                logits = predictor(hidden_states)
                not_cached_experts = self._get_not_cached_experts(
                    next_layer_id,
                    rids,
                    logits.shape[-1],
                )
                self.preds[next_layer_id] = self.topk(
                    hidden_states,
                    logits,
                    not_cached_experts,
                ).topk_ids
            return

        if next_layer_id in self.gates:
            num_experts = self.gates[next_layer_id].output_size
            not_cached_experts = self._get_not_cached_experts(
                next_layer_id,
                rids,
                num_experts,
            )

            next_gate = self.gates[next_layer_id]
            router_logits, _ = next_gate(hidden_states)
            self.preds[next_layer_id] = self.topk(hidden_states, router_logits, not_cached_experts).topk_ids

    def get_pred(self, layer_id):
        if not self.is_enabled: return None

        if layer_id in self.preds:
            return self.preds[layer_id].tolist()
        else:
            return None

global early_gate
early_gate = EarlyGate()

class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
        is_nextn: bool = False,
        support_shared_expert_fusion: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        self.num_experts = config.num_experts
        self.num_shared_experts = 0
        self.num_fused_shared_experts = 0
        if hasattr(config, "n_shared_experts"):
            # config defines the number of shared experts
            self.num_shared_experts = config.n_shared_experts
        elif (
            hasattr(config, "shared_expert_intermediate_size")
            and config.shared_expert_intermediate_size > 0
        ):
            # n_shared_experts is not defined, but shared_expert_intermediate_size is defined, so we use 1 as the number of shared experts
            self.num_shared_experts = 1

        self.enable_shared_expert_fusion = False  # default to False
        if _use_aiter:
            # enable shared expert fusion when use aiter
            self.enable_shared_expert_fusion = (
                support_shared_expert_fusion
                and can_fuse_shared_expert(config, quant_config)
            )
        if self.enable_shared_expert_fusion:
            self.num_fused_shared_experts = self.num_shared_experts

        if exp_args["topk_method"] == "vanilla":
            self.topk = TopK(
                top_k=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                layer_id=layer_id,
            )
        elif exp_args["topk_method"] == "cache_aware":
            self.topk = CacheAwareTopk(
                top_k=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                layer_id=layer_id,
                fixed_num_experts=exp_args["fixed_num_experts"]
            )
        else:
            raise NotImplementedError

        self.cache = {}

        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=(
                config.num_experts_per_tok
                if not self.enable_shared_expert_fusion
                else config.num_experts_per_tok + self.num_fused_shared_experts
            ),
            num_experts=(
                config.num_experts + get_global_server_args().ep_num_redundant_experts
                if not self.enable_shared_expert_fusion
                else config.num_experts
                + get_global_server_args().ep_num_redundant_experts
                + self.num_fused_shared_experts
            ),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            num_fused_shared_experts=self.num_fused_shared_experts,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        # The MTP/nextn draft head is read-only and never prefetches, so it must
        # NOT register its gate: early_gate is a global singleton shared with the
        # target model, and the draft's layer_id collides with a target layer's
        # (add_gate would assert). Only target layers register.
        if not is_nextn:
            early_gate.add_gate(layer_id, self.gate)

        # When enable_shared_expert_fusion, the shared expert runs inside the MoE kernel
        # (via _append_shared_to_topk_output); a separate shared_expert MLP would
        # double-count. If fusion is off (num_fused_shared_experts == 0), keep shared_expert.
        if (
            config.shared_expert_intermediate_size > 0
            and not self.enable_shared_expert_fusion
        ):
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    else {}
                ),
            )
        else:
            self.shared_expert = None
        if _is_cpu and _is_cpu_amx_available:
            self.shared_expert_gate = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=add_prefix("shared_expert_gate", prefix),
            )
        else:
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

        if get_moe_a2a_backend().is_deepep():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.num_experts + get_global_server_args().ep_num_redundant_experts
            )
            self.top_k = config.num_experts_per_tok
        self.is_nextn = is_nextn

    def flush_cache(self, rid):
        if rid in self.cache:
            self.cache[rid].flush()

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def _get_shared_expert_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return sigmoid(shared_expert_gate) for fused shared expert weights."""
        if not self.enable_shared_expert_fusion or self.shared_expert_gate is None:
            return None
        shared_out = self.shared_expert_gate(hidden_states)
        shared_logits = shared_out[0] if isinstance(shared_out, tuple) else shared_out
        return F.sigmoid(shared_logits)

    def _append_shared_to_topk_output(
        self,
        topk_output: StandardTopKOutput,
        hidden_states: torch.Tensor,
    ) -> StandardTopKOutput:
        """Append shared expert ids and weights to topk output before fused MoE."""
        if not self.enable_shared_expert_fusion:
            return topk_output
        shared_weights = self._get_shared_expert_weights(hidden_states)
        if shared_weights is None:
            return topk_output

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
            fused_append_shared_experts_with_weights,
        )

        fused_topk_ids, fused_topk_weights = fused_append_shared_experts_with_weights(
            topk_output.topk_ids,
            topk_output.topk_weights,
            shared_weights,
            self.num_fused_shared_experts,
            N=self.num_experts,
        )
        return StandardTopKOutput(
            topk_weights=fused_topk_weights,
            topk_ids=fused_topk_ids,
            router_logits=topk_output.router_logits,
        )

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                if use_intel_amx_backend(self.shared_expert_gate):
                    shared_output = torch.ops.sgl_kernel.fused_linear_sigmoid_mul(
                        hidden_states,
                        self.shared_expert_gate.weight,
                        self.shared_expert_gate.bias,
                        True,
                        shared_output,
                    )
                else:
                    shared_output = (
                        F.sigmoid(self.shared_expert_gate(hidden_states))
                        * shared_output
                    )

        return shared_output

    def _forward_deepep(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        enable_dual_stream = (
            is_npu()
            and envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            and forward_batch.forward_mode.is_cuda_graph()
        )
        shared_output = None
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            if enable_dual_stream:
                shared_output = shared_expert_on_independent_stream(
                    hidden_states.clone(), self._forward_shared_experts
                )
            else:
                shared_output = self._forward_shared_experts(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=(
                    ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    )
                    if not self.is_nextn
                    else None
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        if enable_dual_stream:
            wait_share_stream()

        if shared_output is not None:
            final_hidden_states.add_(shared_output)

        return final_hidden_states

    def _forward_router_experts(self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None,):
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        num_tokens, n_experts = router_logits.shape
        bs = forward_batch.batch_size

        # The MTP/nextn draft head never touches the TARGET's cache (no prefetch,
        # no target-LRU mutation). Two variants via draft_topk_method:
        #   "vanilla" (default): original MTP, true top-k routing.
        #   "cache_aware": the draft keeps its OWN per-request dynamic cache
        #     (pure dynamic, static_cap=0, NOT in CacheRegistry so its layer_id
        #     can't collide with a target layer) and routes against it, then
        #     updates that own cache with what it selected. Still read-only w.r.t.
        #     the target.
        if self.is_nextn:
            # Needs a dynamic cache to route against (static_cap=0 for the draft).
            draft_cache_aware = (not isinstance(self.topk, TopK)
                                 and exp_args.get("draft_topk_method", "vanilla") == "cache_aware"
                                 and exp_args["cache_dynamic_cap"] > 0)
            cached_experts = None
            tpr = (num_tokens // forward_batch.batch_size) if forward_batch.batch_size else 1
            if draft_cache_aware:
                for rid in forward_batch.rids:
                    if rid not in self.cache:
                        self.cache[rid] = Cache(self.layer_id, rid, self.num_experts,
                                                static_cap=0,
                                                dynamic_cap=exp_args["cache_dynamic_cap"],
                                                record_activations=False)
                cached_experts = [self.cache[rid].get_experts_in_cache()
                                  for rid in forward_batch.rids for _ in range(tpr)]
            if isinstance(self.topk, TopK):
                topk_output = self.topk(hidden_states, router_logits)
            else:
                topk_output = self.topk(hidden_states, router_logits, cached_experts)
            if draft_cache_aware:
                selected = topk_output.topk_ids
                for i, rid in enumerate(forward_batch.rids):
                    for t in range(tpr):
                        self.cache[rid]._update_lru(selected[i * tpr + t].tolist())
            return self.experts(hidden_states, topk_output)

        is_decode = bs == num_tokens
        is_verify = forward_batch.forward_mode.is_target_verify()
        # Decode and target-verify both use cache-aware routing + prefetch +
        # recording + cache updates. A verify step packs tokens_per_req candidate
        # tokens per request, grouped by request along the token axis:
        # [req0_tok0..tokK, req1_tok0..tokK, ...].
        is_step = is_decode or is_verify
        tokens_per_req = (num_tokens // bs) if is_verify else 1

        if is_step:
            # Predict/prefetch once per request. In verify we feed the first
            # candidate token of each request (rows 0, tpr, 2*tpr, ...) so the
            # predictor sees one row per request, matching len(rids).
            pred_hidden = hidden_states[0::tokens_per_req] if is_verify else hidden_states
            early_gate.make_pred(self.layer_id, pred_hidden, forward_batch.rids)
            prefetch_experts = early_gate.get_pred(self.layer_id)
            if prefetch_experts:
                for i, rid in enumerate(forward_batch.rids):
                    self.cache[rid].prefetch(prefetch_experts[i])

            # One cached-expert snapshot per ROW; verify repeats each request's
            # snapshot across its tokens_per_req candidate tokens (frozen snapshot).
            cached_experts = [
                self.cache[rid].get_experts_in_cache()
                for rid in forward_batch.rids
                for _ in range(tokens_per_req)
            ]
        else:
            cached_experts = None

        if isinstance(self.topk, TopK):
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk(hidden_states, router_logits, cached_experts)
        selected = topk_output.topk_ids

        n_tokens = selected.shape[0]

        if is_decode: assert n_tokens == len(forward_batch.rids)
        if is_verify: assert n_tokens == len(forward_batch.rids) * tokens_per_req

        for i, rid in enumerate(forward_batch.rids):
            if rid not in RIDS: RIDS.append(rid)

            if rid not in self.cache:
                record_activations = False if len(RIDS) <= 1 else rid == RIDS[1]
                self.cache[rid] = Cache(self.layer_id, rid, self.num_experts, exp_args["cache_static_cap"], exp_args["cache_dynamic_cap"], record_activations=record_activations)
                CacheRegistry.add(self.cache[rid])

            if is_verify:
                rows = [selected[i * tokens_per_req + t].tolist()
                        for t in range(tokens_per_req)]
                self.cache[rid].read_verify(rows)
            else:
                self.cache[rid].read(selected[i].tolist(), is_decode)

        if self.enable_shared_expert_fusion and TopKOutputChecker.format_is_standard(
            topk_output
        ):
            assert False, "make sure code never reaches here"
            topk_output = self._append_shared_to_topk_output(topk_output, hidden_states)
        return self.experts(hidden_states, topk_output)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(hidden_states.clone())

        with torch.cuda.stream(self.alt_stream):
            router_output = self._forward_router_experts(hidden_states)

        current_stream.wait_stream(self.alt_stream)

        return router_output, shared_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if get_moe_a2a_backend().is_deepep():
            return self._forward_deepep(hidden_states, forward_batch)

        if (
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states
            )
        else:
            shared_output = self._forward_shared_experts(hidden_states)
            final_hidden_states = self._forward_router_experts(hidden_states, forward_batch)

        if shared_output is not None:
            # In-place add is required to keep final_hidden_states in the
            # symmetric memory pool (when --enable-symm-mem is used).
            # An out-of-place add would allocate a new tensor outside symm
            # memory, breaking subsequent symmetric collective operations.
            final_hidden_states += shared_output
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class Qwen2MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        qkv_bias: int = True,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta, rope_scaling = get_rope_config(config)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        qkv_bias = getattr(config, "qkv_bias", True)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qkv_bias=qkv_bias,
            prefix=add_prefix("self_attn", prefix),
        )

        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Qwen2MoE all layers are sparse and have no nextn now
        self.is_layer_sparse = True
        is_previous_layer_sparse = True
        is_next_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
                **kwargs,
            )
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class Qwen2MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2MoeDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        self.moe_dp_size = get_moe_data_parallel_world_size()
        self.attn_cp_size = get_attn_context_model_parallel_world_size()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2MoeDecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2MoeDecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        if (
            is_prefill_context_parallel_enabled()
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        aux_hidden_states = []
        if forward_batch.can_run_tbo:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                ctx = (
                    nullcontext()
                    if not get_global_server_args().disable_piecewise_cuda_graph
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                        captured_last_layer_outputs=(
                            aux_hidden_states
                            if getattr(layer, "_is_layer_to_capture", False)
                            else None
                        ),
                    )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if (
            self.pp_group.is_last_rank
            and is_prefill_context_parallel_enabled()
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.attn_cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Qwen2MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.model = Qwen2MoeModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            alt_stream=alt_stream,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

        # decoder layer
        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.set_eagle3_layers_to_capture(
                [
                    2,
                    num_layers // 2,
                    num_layers - 3,
                ]
            )  # Specific layers for EAGLE3 support
        else:
            self.model.set_eagle3_layers_to_capture([val + 1 for val in layer_ids])


EntryClass = Qwen2MoeForCausalLM
