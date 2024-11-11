from dataclasses import dataclass

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from flux.modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        yield ("pre_img_in", dict(img=img))

        # running on sequences img
        img = self.img_in(img)
        time_emb = timestep_embedding(timesteps, 256)
        yield ("pre_basic_vec", dict(y=y, time_emb=time_emb))
        vec = self.time_in(time_emb)
        yield ("basic_vec", dict(vec=vec))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            g_emb = timestep_embedding(guidance, 256)
            yield ("guidance_emb", dict(g_emb=g_emb, guidance=guidance))
            vec = vec + self.guidance_in(g_emb)
        yield ("guidance_vec", dict(vec=vec))
        vec = vec + self.vector_in(y)
        yield ("vec", dict(vec=vec))
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)


        yield ("pre_double", dict(img=img, txt=txt, vec=vec, pe=pe))

        for i, block in enumerate(tqdm(
            self.double_blocks
            # []
            )):
            for tag, data in block(img=img, txt=txt, vec=vec, pe=pe):
                if i == 0:
                    t = tag
                    if t:
                        t = "_" + t
                    yield ("first_double" + t, data)
                if tag == "":
                    img, txt = data["img"], data["txt"]
                t = tag
                if t:
                    t = "." + t
                yield (f"double_block.{i}{t}", data)


        img = torch.cat((txt, img), 1)
        yield ("pre_single", dict(data=img))
        
        for i, block in enumerate(tqdm(self.single_blocks)):
            for tag, data in block(img, vec=vec, pe=pe):
                if i == 0:
                    t = tag
                    if t:
                        t = "_" + t
                    yield ("first_single" + t, data)
                if tag == "":
                    img = data["img"]
                t = tag
                if t:
                    t = "." + t
                yield (f"single_block.{i}{t}", data)
        img = img[:, txt.shape[1] :, ...]
        
        yield ("pre_final", dict(data=img))
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        # return img
        yield ("final_img", dict(img=img))
