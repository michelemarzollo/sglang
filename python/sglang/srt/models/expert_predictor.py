"""Trained MoE expert-activation predictors.

Vendored, inference-only copy of the predictor model definitions from the
`moe-activation-predictor` repo (`data/model.py`). Used by the `EarlyGate` in
`qwen2_moe.py` to predict the experts a future layer will activate, replacing
the "run the future layer's gate on the current hidden states" heuristic.

Checkpoints are produced by `moe-activation-predictor/train/trainer.py` and laid
out as:

    {ckpt_root}/layer{T}/layer{T-offset}_moe_hidden_states/latest/best.pt
    {ckpt_root}/layer{T}/layer{T-offset}_moe_hidden_states/latest/config.json

where T is the *target* layer whose experts we predict and the input is the MoE
block input (`moe_hidden_states`) of layer T-offset. Each `best.pt` holds
`{"model_state_dict", "config", ...}`; the sibling `config.json` mirrors the
config so the model can be rebuilt without unpickling the full checkpoint.
"""

import json
import os

import torch


def apply_rmsnorm(x, weight, eps):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight


class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.output_dim = config["output_dim"]
        self.dropout = config["dropout"]

        self.dtype = torch.float32
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            torch.nn.Linear(self.input_dim, self.hidden_dims[0], dtype=self.dtype)
        )
        self.layers.append(torch.nn.BatchNorm1d(self.hidden_dims[0]))
        self.layers.append(torch.nn.GELU())
        self.layers.append(torch.nn.Dropout(self.dropout))

        for l in range(len(self.hidden_dims) - 1):
            self.layers.append(
                torch.nn.Linear(
                    self.hidden_dims[l], self.hidden_dims[l + 1], dtype=self.dtype
                )
            )
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(self.dropout))

        self.layers.append(
            torch.nn.Linear(self.hidden_dims[-1], self.output_dim, dtype=self.dtype)
        )

    def forward(self, x):
        x = x.to(self.dtype)
        for layer in self.layers:
            x = layer(x)
        return x


class GatePredictor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.use_attn_norm = config.get("use_attn_norm", False)
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-6)

        self.dtype = torch.float32
        self.gate = torch.nn.Linear(
            self.input_dim, self.output_dim, bias=False, dtype=self.dtype
        )

        if self.use_attn_norm:
            self.post_attention_layernorm_weight = torch.nn.Parameter(
                torch.ones(self.input_dim, dtype=self.dtype)
            )
        else:
            self.register_parameter("post_attention_layernorm_weight", None)

    def forward(self, x):
        x = x.to(self.dtype)
        if self.post_attention_layernorm_weight is not None:
            x = apply_rmsnorm(
                x, self.post_attention_layernorm_weight, self.rms_norm_eps
            )
        return self.gate(x)


def build_model(config):
    model_type = config.get("model_type", "mlp")
    if model_type == "mlp":
        return MLP(config)
    if model_type in {"gate", "early_gate"}:
        return GatePredictor(config)
    raise ValueError(f"Unsupported model_type: {model_type}")


def predictor_dir(ckpt_root, target_layer, offset):
    """Directory holding the predictor for target layer T given input offset."""
    return os.path.join(
        ckpt_root,
        f"layer{int(target_layer)}",
        f"layer{int(target_layer) - int(offset)}_moe_hidden_states",
        "latest",
    )


def load_predictor(ckpt_root, target_layer, offset, device, ckpt_name="best.pt"):
    """Build and load the predictor for `target_layer` (input from T-offset).

    Returns an eval-mode, frozen model on `device`, or None if no checkpoint
    exists for this (target_layer, offset) pair.
    """
    base = predictor_dir(ckpt_root, target_layer, offset)
    ckpt_path = os.path.join(base, ckpt_name)
    config_path = os.path.join(base, "config.json")
    if not os.path.exists(ckpt_path):
        return None

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = checkpoint["config"]

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model
