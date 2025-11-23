from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class TinyReInjectionModel_Carry:
    """Minimal carry structure for non-ACT model. Only stores labels for loss computation."""
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TinyReInjectionModel_Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Structure config (compatible with trm.yaml naming)
    H_cycles: int  # Number of high-level cycles (number of modules in the model)
    L_cycles: int  # Number of low-level cycles per high-level cycle (number of layers in a module)
    L_layers: int  # Number of transformer layers (number of blocks in a layer with input reinjection)
    H_layers: int = 0  # Ignored, for compatibility

    # # Flattened structure config
    # N_supervision: int  # Number of supervision steps
    # T: int  # Number of passes per supervision step
    # n: int  # Number of blocks with input reinjection

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    forward_dtype: str = "bfloat16"
    puzzle_emb_len: int = 16  # if non-zero, its specified to this value
    
    # ACT config (ignored, for compatibility)
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0
    no_ACT_continue: bool = True


class TinyReInjectionModelBlock(nn.Module):
    """Block that processes y and z states, sharing the same layers like TRM."""
    def __init__(self, config: TinyReInjectionModel_Config) -> None:
        super().__init__()
        self.config = config

        # Shared attention (like L_level in TRM)
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        
        # Shared MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cos_sin: Rotary embedding cos/sin
            y: y state [B, L, D]
            z: z state [B, L, D]
            input_injection: Optional input to inject (x) [B, L, D]
        
        Returns:
            hidden_states: Updated states
        
        Pattern matches TRM:
        - z_L = L_level(z_L, z_H + input_embeddings)
        - z_H = L_level(z_H, z_L)
        So: y uses z + x as injection, z uses y as injection
        """

        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        
        # Process z: z + y (using the UPDATED y, like z_H = L_level(z_H, z_L) uses updated z_L)
        # Following TRM pattern: z_H = L_level(z_H, z_L)
        # z = z + y
        # z = rms_norm(z + self.self_attn(cos_sin=cos_sin, hidden_states=z), variance_epsilon=self.norm_eps)
        # z = rms_norm(z + self.mlp(z), variance_epsilon=self.norm_eps)
        
        return hidden_states

# class TinyReInjectionModelLayer(nn.Module):
#     def __init__(self, config: TinyReInjectionModel_Config):
#         super().__init__()
#         self.config = config
#         self.layers = nn.ModuleList([TinyReInjectionModelBlock(config) for _ in range(config.l_layers)])
    
#     def forward(self, cos_sin: CosSin, y: torch.Tensor, z: torch.Tensor, input_injection: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         z = y+z
#         if input_injection is not None:
#             z = z + input_injection
#         for layer in self.layers:
#             z = layer(cos_sin=cos_sin, hidden_states=z)
#         return z

class TinyReInjectionModelLayer(nn.Module):
    def __init__(self, config: TinyReInjectionModel_Config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([TinyReInjectionModelBlock(config) for _ in range(config.L_layers)])
    
    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, input_injection: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_injection is not None:
            hidden_states = hidden_states + input_injection
        for block in self.blocks:
            hidden_states = block(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states

class TinyReInjectionModule(nn.Module):
    def __init__(self, config: TinyReInjectionModel_Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TinyReInjectionModelLayer(config) for _ in range(config.L_cycles)])
        self.final_layer = TinyReInjectionModelLayer(config)
    
    def forward(self, cos_sin: CosSin, y: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = y + z 

        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states, input_injection=x)
        
        z = hidden_states
        y = self.final_layer(cos_sin=cos_sin, hidden_states=hidden_states, input_injection=None)

        return y, z


class TinyReInjectionModel(nn.Module):
    """Fully flattened structure with no recursion, no special gradient pass, no early stopping."""
    
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyReInjectionModel_Config(**config_dict)
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, 
            self.config.hidden_size, 
            init_std=embed_init_std, 
            cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size) 
            if self.config.puzzle_emb_len == 0 
            else self.config.puzzle_emb_len
        )  # ceil div
        
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers, 
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size, 
                init_std=0, 
                cast_to=self.forward_dtype
            )

        # Position encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len, 
                self.config.hidden_size, 
                init_std=embed_init_std, 
                cast_to=self.forward_dtype
            )
        else:
            self.rotary_emb = None

        # Initial states
        self.y_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), 
            persistent=True
        )
        self.z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), 
            persistent=True
        )

        self.modules = nn.ModuleList([TinyReInjectionModule(self.config) for _ in range(self.config.H_cycles)])
        
        # Dummy Q-head for compatibility (not used without ACT)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    @property
    def puzzle_emb(self):
        # puzzle_emb is only created if puzzle_emb_ndim > 0
        # Access the underlying attribute from __dict__ to avoid recursion
        return self.__dict__.get('puzzle_emb', None)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Create input embeddings from tokens and puzzle identifiers."""
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), 
                dim=-2
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            # embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
            embedding = 1 / math.sqrt(2) * embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)

        # Scale
        return self.embed_scale * embedding

    def _get_cos_sin(self):
        """Get rotary embedding cos/sin if available."""
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            return self.rotary_emb()
        return None

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Create initial carry with labels. All sequences are 'halted' since no ACT."""
        batch_size = batch["inputs"].shape[0]
        return TinyReInjectionModel_Carry(
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # All halted (no ACT)
            current_data={k: v.clone() for k, v in batch.items()}
        )

    def forward(self, carry: TinyReInjectionModel_Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyReInjectionModel_Carry, Dict[str, torch.Tensor]]:
        """
        Forward pass with flattened structure:
        - N_supervision outer loop
        - T passes per supervision step
        - n blocks with input reinjection
        - 1 final block without input reinjection
        """
        batch_size = batch["inputs"].shape[0]
        seq_info = dict(cos_sin=self._get_cos_sin())

        # Input embedding (x)
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        
        # Initialize y and z states
        y = self.y_init.expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1)
        z = self.z_init.expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1)

        # y, z = self.blocks_with_reinjection[i](**seq_info, y=y, z=z, input_injection=x)
        
        # final block: no input reinjection
        # y, z = self.final_block(**seq_info, y=y, z=z, input_injection=None)


        for module in self.modules:
            y, z = module(cos_sin=seq_info["cos_sin"], y=y, z=z, x=x)

        # Output heads
        y_hat = self.lm_head(y)[:, self.puzzle_emb_len:]  # Remove puzzle embedding prefix
        
        # Dummy Q-head logits for compatibility (not used without ACT)
        q_logits = self.q_head(y[:, 0]).to(torch.float32)  # Use first puzzle_emb position
        
        # Update carry: keep labels, mark all as halted (single forward pass)
        new_carry = TinyReInjectionModel_Carry(
            steps=torch.ones((batch_size,), dtype=torch.int32),  # Single step
            halted=torch.ones((batch_size,), dtype=torch.bool),  # All halted
            current_data={"labels": batch["labels"]}  # Keep labels for loss
        )
        
        outputs = {
            "logits": y_hat,
            "q_halt_logits": q_logits[..., 0],  # Dummy values for compatibility
            "q_continue_logits": q_logits[..., 1],  # Dummy values for compatibility
        }

        return new_carry, outputs
        

