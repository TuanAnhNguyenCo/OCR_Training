from .mvitv2_utils import *


def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int) :
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim

class MViT(nn.Module):
    def __init__(
        self,
        spatial_size: Tuple[int, int],
        block_setting: Sequence[MSBlockConfig],
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        proj_after_attn: bool,
        dropout: float = 0.5,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: int = 400,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        patch_embed_kernel: Tuple[int, int, int] = (7, 7),
        patch_embed_stride: Tuple[int, int, int] = (4, 4),
        patch_embed_padding: Tuple[int, int, int] = (3, 3),
        input_channels = None,
        output_channels = None,
        **kwargs
    ) :
        """
        MViT main class.

        Args:
            spatial_size (tuple of ints): The spacial size of the input as ``(H, W)``.
            temporal_size (int): The temporal size ``T`` of the input.
            block_setting (sequence of MSBlockConfig): The Network structure.
            residual_pool (bool): If True, use MViTv2 pooling residual connection.
            residual_with_cls_embed (bool): If True, the addition on the residual connection will include
                the class embedding.
            rel_pos_embed (bool): If True, use MViTv2's relative positional embeddings.
            proj_after_attn (bool): If True, apply the projection after the attention.
            dropout (float): Dropout rate. Default: 0.0.
            attention_dropout (float): Attention dropout rate. Default: 0.0.
            stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
            num_classes (int): The number of classes.
            block (callable, optional): Module specifying the layer which consists of the attention and mlp.
            norm_layer (callable, optional): Module specifying the normalization layer to use.
            patch_embed_kernel (tuple of ints): The kernel of the convolution that patchifies the input.
            patch_embed_stride (tuple of ints): The stride of the convolution that patchifies the input.
            patch_embed_padding (tuple of ints): The padding of the convolution that patchifies the input.
        """
        super().__init__()
        print("Load Model: MViT_V2")
        # This implementation employs a different parameterization scheme than the one used at PyTorch Video:
        # https://github.com/facebookresearch/pytorchvideo/blob/718d0a4/pytorchvideo/models/vision_transformers.py
        # We remove any experimental configuration that didn't make it to the final variants of the models. To represent
        # the configuration of the architecture we use the simplified form suggested at Table 1 of the paper.
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")

        if block is None:
            block = MultiscaleBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Patch Embedding module
        self.conv_proj = nn.Conv2d(
            in_channels=3,
            out_channels=block_setting[0].input_channels,
            kernel_size=patch_embed_kernel,
            stride=patch_embed_stride,
            padding=patch_embed_padding,
        )

        input_size = [size // stride for size, stride in zip( spatial_size, self.conv_proj.stride)]

        # Spatio-Temporal Class Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_size=block_setting[0].input_channels,
            spatial_size=(input_size[0], input_size[1]),
            rel_pos_embed=rel_pos_embed,
        )

        # Encoder module
        self.blocks = nn.ModuleList()
        for stage_block_id, cnf in enumerate(block_setting):
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)

            self.blocks.append(
                block(
                    input_size=input_size,
                    cnf=cnf,
                    residual_pool=residual_pool,
                    residual_with_cls_embed=residual_with_cls_embed,
                    rel_pos_embed=rel_pos_embed,
                    proj_after_attn=proj_after_attn,
                    dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                )
            )

            if len(cnf.stride_q) > 0:
                input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
        print("Num Classes",num_classes)
        self.num_classes = num_classes
        self.dropout = dropout
        self.block_setting = block_setting

        self.in_channels = input_channels
        self.out_channels_ = output_channels
        self.out_channels = sorted(list(set(output_channels)))
       
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        x = clip    
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # Ensure x is contiguous before the loop
        x = x.contiguous()  
        output = []
        thw = self.pos_encoding.spatial_size
        for idx, block in enumerate(self.blocks):
            if self.in_channels[idx] != self.out_channels_[idx]:
                # Make sliced tensor contiguous before view
                sliced_x = x[:, 1:].contiguous()  
                output.append(sliced_x.view(x.shape[0], thw[0], thw[1], self.in_channels[idx]).permute(0, 3, 1, 2))
            x, thw = block(x, thw)

        # Make sliced tensor contiguous before view
        sliced_x = x[:, 1:].contiguous() 
        output.append(sliced_x.view(x.shape[0], thw[0], thw[1], self.out_channels[-1]).permute(0, 3, 1, 2))

        return output




def mvit_v2_s(**kwargs):
    config: Dict[str, List] = {
        "num_heads": [ 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8],
        "input_channels": [32, 32, 32, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256],
        "output_channels": [32, 32, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256],
        "kernel_q": [
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
        ],
        "kernel_kv": [
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
            [ 3, 3],
        ],
        "stride_q": [
            [ 1, 1],
            [ 1, 1],
            [ 2, 2],
            [ 1, 1],
            [ 1, 1],
            [ 2, 2],
            [ 1, 1],
            [ 1, 1],
            [ 1, 1],
            [ 1, 1],
            [ 1, 1],
            [ 1, 1],
            [ 1, 1],
            [ 1, 1],
            [ 1, 1],
            [ 1, 1],
            [ 2, 2],
            [ 1, 1],
        ],
        "stride_kv": [
            [ 8, 8],
            [ 8, 8],
            [ 4, 4],
            [ 4, 4],
            [ 4, 4],
            [ 2, 2],
            [ 2, 2],
            [ 2, 2],
            [ 2, 2],
            [ 2, 2],
            [ 2, 2],
            [ 2, 2],
            [ 2, 2],
            [ 2, 2],
            [ 2, 2],
            [ 2, 2],
            [ 1, 1],
            [ 1, 1],
        ],
    }

    print(len(config["num_heads"]))
    print(len(config["input_channels"]))
    print(len(config["output_channels"]))

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )
    
    return MViT(
        spatial_size=(640, 640),
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        num_classes = kwargs.pop('num_classes',400),
        input_channels=config["input_channels"],
        output_channels=config["output_channels"],
        **kwargs,
    )

if __name__ == "__main__":
    model = mvit_v2_s()
    import torch
    a = torch.rand(2,3,640,640)
    model(a)