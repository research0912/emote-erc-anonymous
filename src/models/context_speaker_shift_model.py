
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType


class ContextSpeakerShiftERC(nn.Module):
    """ERC model with four interpretable evidence streams:
    1. contextual transformer representation,
    2. BiLSTM dialogue-flow representation,
    3. same-speaker latent speaker-state representation,
    4. previous-emotion shift representation.
    """
    def __init__(self, cfg: Dict, num_labels: int):
        super().__init__()
        self.cfg = cfg
        self.num_labels = num_labels
        model_cfg = cfg["model"]
        self.encoder = AutoModel.from_pretrained(model_cfg["base_model"])
        if model_cfg.get("gradient_checkpointing", False) and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        hidden = self.encoder.config.hidden_size

        if model_cfg.get("use_lora", True):
            lora_cfg = LoraConfig(
                r=model_cfg.get("lora_r", 8),
                lora_alpha=model_cfg.get("lora_alpha", 16),
                target_modules=model_cfg.get("lora_target_modules", ["query", "value"]),
                lora_dropout=model_cfg.get("lora_dropout", 0.1),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)

        self.use_bilstm = model_cfg.get("use_bilstm", True)
        self.use_gnn = model_cfg.get("use_gnn", False)
        self.use_speaker_state = model_cfg.get("use_speaker_state", True)
        self.use_emotion_shift = model_cfg.get("use_emotion_shift", True)
        self.use_cross_attention = model_cfg.get("use_cross_attention", False)
        self.dialogue_flow_source = model_cfg.get("dialogue_flow_source", "token")
        self.pooling_strategy = model_cfg.get("pooling_strategy", "cls")
        dropout = model_cfg.get("dropout", 0.2)

        fusion_dim = hidden
        if self.use_bilstm:
            lstm_hidden = model_cfg.get("lstm_hidden_size", 256)
            self.dialogue_bilstm = nn.LSTM(
                input_size=hidden,
                hidden_size=lstm_hidden,
                num_layers=model_cfg.get("lstm_layers", 1),
                batch_first=True,
                bidirectional=True,
            )
            self.dialogue_flow_dim = 2 * lstm_hidden
            fusion_dim += self.dialogue_flow_dim
        else:
            self.dialogue_flow_dim = 0

        if self.use_gnn:
            gnn_hidden = model_cfg.get("gnn_hidden_size", hidden)
            gnn_layers = model_cfg.get("gnn_layers", 2)
            self.gnn_self_weight = model_cfg.get("gnn_self_weight", 1.0)
            self.gnn_adjacent_weight = model_cfg.get("gnn_adjacent_weight", 1.0)
            self.gnn_same_speaker_weight = model_cfg.get("gnn_same_speaker_weight", 1.0)
            self.gnn_input_proj = nn.Sequential(
                nn.Linear(hidden, gnn_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gnn_self_layers = nn.ModuleList([
                nn.Linear(gnn_hidden, gnn_hidden) for _ in range(gnn_layers)
            ])
            self.gnn_neigh_layers = nn.ModuleList([
                nn.Linear(gnn_hidden, gnn_hidden) for _ in range(gnn_layers)
            ])
            self.gnn_norm = nn.LayerNorm(gnn_hidden)
            self.gnn_dropout = nn.Dropout(dropout)
            self.gnn_hidden_dim = gnn_hidden
            fusion_dim += gnn_hidden
        else:
            self.gnn_hidden_dim = 0

        if self.use_cross_attention:
            num_heads = model_cfg.get("cross_attention_heads", 4)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            # stability: proj + layernorm + residual + dropout
            self.cross_attention_proj = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.cross_attention_ln = nn.LayerNorm(hidden)
        else:
            self.cross_attention = None

        if self.use_speaker_state:
            speaker_dim = model_cfg.get("speaker_state_dim", 256)
            self.speaker_state_projector = nn.Sequential(
                nn.Linear(hidden, speaker_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
            self.speaker_state_dim = speaker_dim
            fusion_dim += speaker_dim
        else:
            self.speaker_state_dim = 0

        if self.use_emotion_shift:
            shift_dim = model_cfg.get("emotion_shift_dim", 64)
            self.emotion_shift_source = model_cfg.get("emotion_shift_source", "previous_text")
            self.reuse_turn_repr_for_previous = model_cfg.get("reuse_turn_repr_for_previous", False)
            if self.emotion_shift_source == "gold_label":
                # +1 is used for no previous emotion. This is an oracle setting.
                self.prev_emotion_embedding = nn.Embedding(num_labels + 1, shift_dim)
                shift_input_dim = shift_dim
            else:
                self.start_previous_turn = nn.Parameter(torch.zeros(hidden))
                self.shift_use_product = model_cfg.get("shift_use_product", False)
                shift_input_dim = hidden * (4 if self.shift_use_product else 3)
                # Add cross-attention dimension if enabled
                if self.use_cross_attention:
                    shift_input_dim += hidden
            self.shift_projector = nn.Sequential(
                nn.Linear(shift_input_dim, shift_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.shift_dim = shift_dim
            fusion_dim += shift_dim
        else:
            self.shift_dim = 0

        self.use_attention_fusion = model_cfg.get("use_attention_fusion", False)
        self.attention_fusion_dim = model_cfg.get("attention_fusion_dim", hidden)

        fusion_gate_dim = fusion_dim + (self.attention_fusion_dim if self.use_attention_fusion else 0)
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_gate_dim, fusion_gate_dim),
            nn.Sigmoid(),
        )
        # dynamic fusion option: compute weights for each evidence stream
        self.use_dynamic_fusion = model_cfg.get("use_dynamic_fusion", False)
        if self.use_dynamic_fusion:
            # number of streams: contextual + dialogue_flow (optional) + graph (optional) + speaker_state (optional) + shift (optional)
            self.num_streams = 1 + (1 if self.use_bilstm else 0) + (1 if self.use_gnn else 0) + (1 if self.use_speaker_state else 0) + (1 if self.use_emotion_shift else 0)
            self.fusion_weight_net = nn.Sequential(
                nn.Linear(fusion_dim, max(64, fusion_dim // 4)),
                nn.ReLU(),
                nn.Linear(max(64, fusion_dim // 4), self.num_streams),
            )
        if self.use_attention_fusion:
            self.context_fusion_proj = nn.Sequential(
                nn.Linear(hidden, self.attention_fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.flow_fusion_proj = nn.Sequential(
                nn.Linear(self.dialogue_flow_dim, self.attention_fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) if self.use_bilstm else None
            self.gnn_fusion_proj = nn.Sequential(
                nn.Linear(self.gnn_hidden_dim, self.attention_fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) if self.use_gnn else None
            self.speaker_fusion_proj = nn.Sequential(
                nn.Linear(self.speaker_state_dim, self.attention_fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) if self.use_speaker_state else None
            self.shift_fusion_proj = nn.Sequential(
                nn.Linear(self.shift_dim, self.attention_fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) if self.use_emotion_shift else None
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=self.attention_fusion_dim,
                num_heads=model_cfg.get("attention_fusion_heads", 4),
                dropout=dropout,
                batch_first=True,
            )
            self.fusion_attention_norm = nn.LayerNorm(self.attention_fusion_dim)
        fusion_hidden = model_cfg.get("fusion_hidden_size", 256)
        classifier_input_dim = fusion_dim + (self.attention_fusion_dim if self.use_attention_fusion else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_labels),
        )
        
        # Monte-Carlo Dropout / uncertainty options
        self.use_mc_dropout = model_cfg.get("use_mc_dropout", False)
        self.mc_dropout_samples = model_cfg.get("mc_dropout_samples", 10)

        # Contrastive projection head
        self.use_contrastive = model_cfg.get("use_contrastive", False)
        if self.use_contrastive:
            proj_dim = model_cfg.get("contrastive_proj_dim", 128)
            self.projection_head = nn.Sequential(
                nn.Linear(fusion_gate_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            )
            self.contrastive_temperature = model_cfg.get("contrastive_temperature", 0.07)

    def _mean_pool(self, token_repr: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (token_repr * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _pool_encoder_output(self, enc_out, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "mean":
            return self._mean_pool(enc_out.last_hidden_state, attention_mask)
        if self.pooling_strategy in {"cls_last4_mean", "last4_cls_mean"}:
            hidden_states = getattr(enc_out, "hidden_states", None)
            if hidden_states is not None and len(hidden_states) >= 4:
                return torch.stack([state[:, 0, :] for state in hidden_states[-4:]], dim=0).mean(dim=0)
        return enc_out.last_hidden_state[:, 0, :]

    def _module_dtype(self, module: nn.Module) -> torch.dtype:
        return next(module.parameters()).dtype

    def _dialogue_flow(self, token_repr: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).detach().cpu().clamp_min(1)
        lstm_dtype = self._module_dtype(self.dialogue_bilstm)
        lstm_input = token_repr.to(dtype=lstm_dtype)
        packed = nn.utils.rnn.pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.dialogue_bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        mask = attention_mask[:, :lstm_out.size(1)].unsqueeze(-1).float()
        return (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _cross_attention_pool(
        self,
        query_repr: torch.Tensor,
        context_repr: torch.Tensor,
        context_turn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention: current utterance queries context turns."""
        if self.cross_attention is None or context_repr is None:
            return None
        batch_size, num_turns, hidden = context_repr.shape
        query = query_repr.unsqueeze(1)  # [B, 1, H]
        ca_dtype = self._module_dtype(self.cross_attention)
        query = query.to(dtype=ca_dtype)
        context_repr = context_repr.to(dtype=ca_dtype)
        
        # Create key_padding_mask from context_turn_mask
        key_padding_mask = (context_turn_mask == 0) if context_turn_mask is not None else None
        attended, weights = self.cross_attention(
            query, context_repr, context_repr,
            key_padding_mask=key_padding_mask,
        )  # [B, 1, H]
        attended = attended.squeeze(1)  # [B, H]
        attended = self.cross_attention_proj(attended)
        return attended

    def _encode_pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.pooling_strategy in {"cls_last4_mean", "last4_cls_mean"},
            return_dict=True,
        )
        return self._pool_encoder_output(out, attention_mask)

    def _encode_context_turns(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_turns, seq_len = context_input_ids.shape
        flat_ids = context_input_ids.view(batch_size * num_turns, seq_len)
        flat_mask = context_attention_mask.view(batch_size * num_turns, seq_len)
        return self._encode_pooled(flat_ids, flat_mask).view(batch_size, num_turns, -1)

    def _turn_level_dialogue_flow(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        context_turn_mask: torch.Tensor,
        current_turn_indices: torch.Tensor,
        precomputed_turn_repr: Optional[torch.Tensor] = None,
        ablation: Optional[str] = None,
    ) -> torch.Tensor:
        if precomputed_turn_repr is None:
            turn_repr = self._encode_context_turns(context_input_ids, context_attention_mask)
        else:
            turn_repr = precomputed_turn_repr
        batch_size, num_turns = turn_repr.shape[:2]

        if ablation == "no_context":
            gather_idx = current_turn_indices.to(turn_repr.device).view(batch_size, 1, 1).expand(-1, 1, turn_repr.size(-1))
            turn_repr = turn_repr.gather(1, gather_idx)
            num_turns = 1
            current_turn_indices = torch.zeros(batch_size, dtype=torch.long, device=turn_repr.device)
            turn_mask = torch.ones(batch_size, 1, dtype=context_turn_mask.dtype, device=context_turn_mask.device)
        else:
            turn_mask = context_turn_mask

        lengths = turn_mask.sum(dim=1).detach().cpu().long().clamp_min(1)
        lstm_dtype = self._module_dtype(self.dialogue_bilstm)
        lstm_input = turn_repr.to(dtype=lstm_dtype)
        packed = nn.utils.rnn.pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.dialogue_bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=num_turns)
        gather_idx = current_turn_indices.to(lstm_out.device).view(batch_size, 1, 1).expand(-1, 1, lstm_out.size(-1))
        return lstm_out.gather(1, gather_idx).squeeze(1)

    def _build_turn_graph(
        self,
        context_turn_mask: torch.Tensor,
        context_turn_speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build a dense turn graph with self-loops, adjacent-turn edges, and same-speaker edges."""
        batch_size, num_turns = context_turn_mask.shape
        device = context_turn_mask.device
        valid = context_turn_mask.float()
        adjacency = torch.zeros(batch_size, num_turns, num_turns, device=device, dtype=torch.float32)

        idx = torch.arange(num_turns, device=device)
        adjacency[:, idx, idx] = self.gnn_self_weight
        if num_turns > 1:
            adjacency[:, idx[:-1], idx[1:]] = self.gnn_adjacent_weight
            adjacency[:, idx[1:], idx[:-1]] = self.gnn_adjacent_weight

        if context_turn_speaker_ids is not None:
            same_speaker = context_turn_speaker_ids.unsqueeze(2) == context_turn_speaker_ids.unsqueeze(1)
            adjacency = adjacency + same_speaker.float() * self.gnn_same_speaker_weight

        # mask out padded turns
        adjacency = adjacency * valid.unsqueeze(2) * valid.unsqueeze(1)
        return adjacency

    def _gnn_dialogue_flow(
        self,
        turn_repr: torch.Tensor,
        context_turn_mask: torch.Tensor,
        current_turn_indices: torch.Tensor,
        context_turn_speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """A small graph network over turn embeddings.

        turn_repr: [B, T, H]
        returns current-turn graph embedding: [B, H_gnn]
        """
        h = self.gnn_input_proj(turn_repr)
        adjacency = self._build_turn_graph(context_turn_mask, context_turn_speaker_ids)
        deg = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        norm_adj = adjacency / deg

        for self_layer, neigh_layer in zip(self.gnn_self_layers, self.gnn_neigh_layers):
            neigh = torch.bmm(norm_adj, h)
            h_new = self_layer(h) + neigh_layer(neigh)
            h_new = torch.relu(h_new)
            h_new = self.gnn_dropout(h_new)
            h = self.gnn_norm(h + h_new)

        gather_idx = current_turn_indices.to(h.device).view(turn_repr.size(0), 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(1, gather_idx).squeeze(1)

    def forward(
        self,
        input_ids,
        attention_mask,
        inputs_embeds: Optional[torch.Tensor] = None,
        speaker_input_ids: Optional[torch.Tensor] = None,
        speaker_attention_mask: Optional[torch.Tensor] = None,
        previous_input_ids: Optional[torch.Tensor] = None,
        previous_attention_mask: Optional[torch.Tensor] = None,
        previous_turn_exists: Optional[torch.Tensor] = None,
        context_input_ids: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        context_turn_mask: Optional[torch.Tensor] = None,
        context_turn_speaker_ids: Optional[torch.Tensor] = None,
        current_turn_indices: Optional[torch.Tensor] = None,
        previous_emotion_ids: Optional[torch.Tensor] = None,
        ablation: Optional[str] = None,
    ):
        encoder_kwargs = {"attention_mask": attention_mask}
        if inputs_embeds is not None:
            encoder_kwargs["inputs_embeds"] = inputs_embeds
        else:
            encoder_kwargs["input_ids"] = input_ids
        enc_out = self.encoder(
            **encoder_kwargs,
            output_hidden_states=self.pooling_strategy in {"cls_last4_mean", "last4_cls_mean"},
            return_dict=True,
        )
        token_repr = enc_out.last_hidden_state
        contextual_repr = self._pool_encoder_output(enc_out, attention_mask)
        if ablation == "no_context" and self.dialogue_flow_source != "turn_level":
            contextual_repr = torch.zeros_like(contextual_repr)
        features = [contextual_repr]
        evidence = {"contextual_repr": contextual_repr}

        turn_repr = None
        use_turn_level_context = (
            self.dialogue_flow_source == "turn_level"
            and context_input_ids is not None
            and context_attention_mask is not None
            and context_turn_mask is not None
            and current_turn_indices is not None
        )
        if use_turn_level_context and (
            self.use_bilstm
            or self.use_gnn
            or (self.use_emotion_shift and self.reuse_turn_repr_for_previous)
            or (self.use_emotion_shift and self.use_cross_attention)
        ):
            turn_repr = self._encode_context_turns(context_input_ids, context_attention_mask)

        if self.use_bilstm:
            if ablation in ["no_dialogue_flow", "no_bilstm"]:
                flow_repr = torch.zeros(input_ids.size(0), self.dialogue_flow_dim, device=input_ids.device)
            elif use_turn_level_context:
                flow_repr = self._turn_level_dialogue_flow(
                    context_input_ids,
                    context_attention_mask,
                    context_turn_mask,
                    current_turn_indices,
                    precomputed_turn_repr=turn_repr,
                    ablation=ablation,
                )
            else:
                flow_repr = self._dialogue_flow(token_repr, attention_mask)
            features.append(flow_repr)
            evidence["dialogue_flow_repr"] = flow_repr

        if self.use_gnn:
            if ablation == "no_gnn":
                gnn_repr = torch.zeros(input_ids.size(0), self.gnn_hidden_dim, device=input_ids.device)
            elif use_turn_level_context:
                gnn_repr = self._gnn_dialogue_flow(
                    turn_repr,
                    context_turn_mask,
                    current_turn_indices,
                    context_turn_speaker_ids=context_turn_speaker_ids,
                )
            else:
                gnn_repr = torch.zeros(input_ids.size(0), self.gnn_hidden_dim, device=input_ids.device)
            features.append(gnn_repr)
            evidence["graph_dialogue_repr"] = gnn_repr

        if self.use_speaker_state:
            if ablation == "no_speaker_state" or speaker_input_ids is None:
                speaker_repr = torch.zeros(input_ids.size(0), self.speaker_state_dim, device=input_ids.device)
            else:
                speaker_out = self.encoder(
                    input_ids=speaker_input_ids,
                    attention_mask=speaker_attention_mask,
                    output_hidden_states=self.pooling_strategy in {"cls_last4_mean", "last4_cls_mean"},
                    return_dict=True,
                )
                speaker_pooled = self._pool_encoder_output(speaker_out, speaker_attention_mask)
                speaker_pooled = speaker_pooled.to(dtype=self._module_dtype(self.speaker_state_projector))
                speaker_repr = self.speaker_state_projector(speaker_pooled)
            features.append(speaker_repr)
            evidence["speaker_state_repr"] = speaker_repr

        if self.use_emotion_shift:
            if ablation == "no_shift":
                shift_repr = torch.zeros(input_ids.size(0), self.shift_dim, device=input_ids.device)
            elif self.emotion_shift_source == "gold_label":
                if previous_emotion_ids is None:
                    shift_ids = torch.full((input_ids.size(0),), self.num_labels, dtype=torch.long, device=input_ids.device)
                else:
                    shift_ids = previous_emotion_ids.clone()
                    shift_ids = torch.where(shift_ids < 0, torch.full_like(shift_ids, self.num_labels), shift_ids)
                shift_repr = self.shift_projector(self.prev_emotion_embedding(shift_ids))
            else:
                if previous_input_ids is None or previous_attention_mask is None:
                    previous_pooled = self.start_previous_turn.unsqueeze(0).expand(input_ids.size(0), -1)
                elif self.reuse_turn_repr_for_previous and use_turn_level_context and turn_repr is not None:
                    batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
                    prev_idx = (current_turn_indices.to(input_ids.device) - 1).clamp_min(0)
                    previous_pooled = turn_repr[batch_idx, prev_idx]
                    if previous_turn_exists is not None:
                        exists = previous_turn_exists.to(previous_pooled.device).view(-1, 1).float()
                        start = self.start_previous_turn.unsqueeze(0).expand_as(previous_pooled)
                        previous_pooled = exists * previous_pooled + (1.0 - exists) * start
                else:
                    previous_out = self.encoder(
                        input_ids=previous_input_ids,
                        attention_mask=previous_attention_mask,
                        output_hidden_states=self.pooling_strategy in {"cls_last4_mean", "last4_cls_mean"},
                        return_dict=True,
                    )
                    previous_pooled = self._pool_encoder_output(previous_out, previous_attention_mask)
                    if previous_turn_exists is not None:
                        exists = previous_turn_exists.to(previous_pooled.device).view(-1, 1).float()
                        start = self.start_previous_turn.unsqueeze(0).expand_as(previous_pooled)
                        previous_pooled = exists * previous_pooled + (1.0 - exists) * start
                
                # Compute cross-attention over context turns if available
                cross_attended = None
                if self.use_cross_attention and use_turn_level_context:
                    cross_attended = self._cross_attention_pool(contextual_repr, turn_repr, context_turn_mask)
                
                shift_parts = [previous_pooled, contextual_repr, torch.abs(contextual_repr - previous_pooled)]
                if self.shift_use_product:
                    shift_parts.append(previous_pooled * contextual_repr)
                if cross_attended is not None:
                    shift_parts.append(cross_attended)
                shift_features = torch.cat(shift_parts, dim=-1)
                shift_features = shift_features.to(dtype=self._module_dtype(self.shift_projector))
                shift_repr = self.shift_projector(shift_features)
            features.append(shift_repr)
            evidence["emotion_shift_repr"] = shift_repr

        # Apply stream-level feature ablation if specified
        if ablation in ["no_context", "text"]:
            contextual_repr = torch.zeros_like(contextual_repr)
        
        if ablation in ["no_dialogue_flow", "no_bilstm", "no_gnn", "dialogue_flow"]:
            if self.use_bilstm:
                flow_repr = torch.zeros_like(flow_repr)
            if self.use_gnn:
                gnn_repr = torch.zeros_like(gnn_repr)
                
        if ablation in ["no_speaker_state", "speaker_state"]:
            if self.use_speaker_state:
                speaker_repr = torch.zeros_like(speaker_repr)
                
        if ablation in ["no_shift", "emotion_shift"]:
            if self.use_emotion_shift:
                shift_repr = torch.zeros_like(shift_repr)

        # Re-build features list and streams list using the potentially ablated representations
        features = [contextual_repr]
        streams = [contextual_repr]
        
        if self.use_bilstm:
            features.append(flow_repr)
            streams.append(flow_repr)
        if self.use_gnn:
            features.append(gnn_repr)
            streams.append(gnn_repr)
        if self.use_speaker_state:
            features.append(speaker_repr)
            streams.append(speaker_repr)
        if self.use_emotion_shift:
            features.append(shift_repr)
            streams.append(shift_repr)

        fused = torch.cat(features, dim=-1)
        fused = fused.to(dtype=self._module_dtype(self.fusion_gate))

        if self.use_dynamic_fusion:
            # compute per-stream weights and apply scalar weighting per stream
            weight_logits = self.fusion_weight_net(fused)
            weights = torch.softmax(weight_logits, dim=-1)
            weighted_streams = []
            idx = 0
            for s in streams:
                w = weights[:, idx].unsqueeze(-1)
                weighted_streams.append(s * w)
                idx += 1
            fused = torch.cat(weighted_streams, dim=-1)

        if self.use_attention_fusion:
            attention_streams = [self.context_fusion_proj(contextual_repr)]
            if self.use_bilstm:
                attention_streams.append(self.flow_fusion_proj(flow_repr))
            if self.use_gnn:
                attention_streams.append(self.gnn_fusion_proj(gnn_repr))
            if self.use_speaker_state:
                attention_streams.append(self.speaker_fusion_proj(speaker_repr))
            if self.use_emotion_shift:
                attention_streams.append(self.shift_fusion_proj(shift_repr))

            stream_stack = torch.stack(attention_streams, dim=1)
            fusion_query = self.context_fusion_proj(contextual_repr).unsqueeze(1)
            attended_stream, _ = self.fusion_attention(fusion_query, stream_stack, stream_stack)
            attended_stream = self.fusion_attention_norm(attended_stream.squeeze(1) + self.context_fusion_proj(contextual_repr))
            fused = torch.cat([fused, attended_stream], dim=-1)

        gate_weights = self.fusion_gate(fused)
        gated = fused * gate_weights
        logits = self.classifier(gated)

        return {
            "logits": logits,
            "features": gated,
            "evidence": evidence,
            "fused": fused,
            "gate_weights": gate_weights
        }

    # --- MC-Dropout / Uncertainty helpers ---
    def enable_mc_dropout(self):
        # Put dropout modules into train mode while keeping others in eval
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

    def mc_dropout_predict(self, n_samples: Optional[int] = None, **inputs):
        """Run multiple stochastic forward passes and return mean logits and std (uncertainty).
        Pass the same kwargs as `forward` via `inputs`.
        """
        n = n_samples or self.mc_dropout_samples
        was_training = self.training
        self.eval()
        # enable dropout layers only
        self.enable_mc_dropout()
        logits_list = []
        with torch.no_grad():
            for _ in range(n):
                out = self.forward(**inputs)
                logits_list.append(out["logits"].unsqueeze(0))
        logits_stack = torch.cat(logits_list, dim=0)  # [n, B, C]
        mean_logits = logits_stack.mean(dim=0)
        std_logits = logits_stack.std(dim=0)
        # restore
        self.disable_mc_dropout()
        if was_training:
            self.train()
        return mean_logits, std_logits

    def predictive_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute predictive entropy for probability vectors (B, C)."""
        eps = 1e-12
        ent = - (probs * torch.log(probs + eps)).sum(dim=-1)
        return ent

    def detect_ambiguous(self, mean_logits: torch.Tensor, std_logits: torch.Tensor, threshold: float = 0.5):
        """Return boolean mask of ambiguous utterances based on predictive entropy or std threshold.
        `threshold` applies to predictive entropy by default (in nats)."""
        probs = F.softmax(mean_logits, dim=-1)
        ent = self.predictive_entropy(probs)
        ambiguous = ent > threshold
        return ambiguous, ent

    # --- Contrastive / projection ---
    def compute_contrastive_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent contrastive loss between two batches z_i and z_j (shape N x D)."""
        if not self.use_contrastive:
            return torch.tensor(0.0, device=z_i.device)
        z_i = F.normalize(self.projection_head(z_i), dim=-1)
        z_j = F.normalize(self.projection_head(z_j), dim=-1)
        batch_size = z_i.size(0)
        temperature = self.contrastive_temperature

        # similarity matrix
        logits = (z_i @ z_j.t()) / temperature  # [N, N]
        labels = torch.arange(batch_size, device=z_i.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_i + loss_j)
        return loss

    def get_bilstm_sequence(self, token_repr: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return BiLSTM outputs per token/turn before pooling for visualization/graphing."""
        if not self.use_bilstm:
            return None
        lengths = attention_mask.sum(dim=1).detach().cpu().clamp_min(1)
        lstm_dtype = self._module_dtype(self.dialogue_bilstm)
        lstm_input = token_repr.to(dtype=lstm_dtype)
        packed = nn.utils.rnn.pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.dialogue_bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return lstm_out
