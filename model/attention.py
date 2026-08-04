import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask = None, key_padding_mask = None):
        """
        - query: [batch, heads, query_len, head_dim]
        - key:   [batch, heads, key_len, head_dim]
        - value: [batch, heads, key_len, value_dim]

        - output:       [batch, heads, query_len, value_dim]
        - attn_weights: [batch, heads, query_len, key_len]
        """
        head_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        # apply the attention mask if it is provided
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        # apply the key padding mask if it is provided
        if key_padding_mask is not None:
            padding_mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(padding_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = ScaledDotProductAttention(dropout)
        # linear projections for query, key, and value
        # these are the learnable parameters of the attention mechanism
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask = None, key_padding_mask = None):
        """
        - query: [batch, query_len, embed_dim]
        - key:   [batch, key_len, embed_dim]
        - value: [batch, key_len, embed_dim]

        - output:       [batch, query_len, embed_dim]
        - attn_weights: [batch, query_len, key_len]
        """
        # project the input query, key and value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        # add a fake dimension for multi-head attention compatibility
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        # apply the attention mechanism
        output, attn_weights = self.attention(q, k, v, attn_mask, key_padding_mask)
        output = output.squeeze(1) # remove the fake dimension
        attn_weights = attn_weights.squeeze(1) # remove the fake dimension
        # project the output
        output = self.out_proj(output)
        return output, attn_weights
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention = ScaledDotProductAttention(dropout)
        # linear projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask = None, key_padding_mask = None):
        """
        - query: [batch, query_len, embed_dim]
        - key:   [batch, key_len, embed_dim]
        - value: [batch, key_len, embed_dim]

        - output:       [batch, query_len, embed_dim]
        - attn_weights: [batch, heads, query_len, key_len]
        """
        # project the input query, key and value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        # reshape for multi-head attention
        batch_size = query.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        q = q.transpose(1, 2) # [batch, heads, query_len, head_dim]
        k = k.transpose(1, 2) # [batch, heads, key_len, head_dim]
        v = v.transpose(1, 2) # [batch, heads, key_len, head_dim]
        # apply the attention mechanism
        output, attn_weights = self.attention(q, k, v, attn_mask, key_padding_mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # project the output
        output = self.out_proj(output)
        return output, attn_weights
        

class DeformableAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_points: int = 4, dropout: float = 0.0):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_points = num_points
        # linear projections for query, key, and value
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.Linear(embed_dim, num_heads * num_points)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2) # learnable offsets for deformable attention
        self.dropout = nn.Dropout(dropout)

    def _sample_at_points(self, value, sampling_locations, spatial_shape):
        """
        - value: [batch, heads, H~ * W~, head_dim]
        - sampling_locations: [batch, query_len, heads, num_points, 2]
        - spatial_shape: [H~, W~] (height, width)

        - sampled_value: [batch, query_len, heads, num_points, head_dim]

        We sample points from the value feature map at the locations in sampling_locations.
        For each head, we sample num_points points.
        """
        # value contains the features, we sample at this offset points
        # since the offset_points are normalized to [0, 1], we use grid_sample for sampling
        h, w = spatial_shape
        B, Q, H, P, _ = sampling_locations.size()
        # resize the value
        # grid_sample expects the input to be in the shape of [N, C, H_in, W_in]
        # WARNING: grid_sample input is channel first, but logically it makes more sense to think it for simplicity as [N, H_in, W_in, C]
        #          In anyway we need to make it channel first
        # [B, heads, H~ * W~, head_dim] -> [B, heads, head_dim, H~, W~] -> [B * heads, head_dim, H~, W~]
        value = value.view(B, self.num_heads, h, w, self.head_dim).permute(0, 1, 4, 2, 3).contiguous()
        value = value.view(B * self.num_heads, self.head_dim, h, w)
        # create the grid
        # grid_sample expects the grid to be in the shape of [N, H_out, W_out, 2]
        # [B, Q, heads, num_points, 2] -> [B, heads, Q, num_points, 2] -> [B * heads, Q, num_points, 2]
        grid = sampling_locations.permute(0, 2, 1, 3, 4)
        grid = grid.contiguous().view(B * self.num_heads, Q, self.num_points, 2)
        grid = grid * 2 - 1 # normalize to [-1, 1] for grid_sample
        sampled_values = F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        # sampled_values: [B * heads, head_dim, Q, num_points]
        sampled_values = sampled_values.view(B, self.num_heads, self.head_dim, Q, self.num_points)
        sampled_values = sampled_values.permute(0, 3, 1, 4, 2).contiguous()
        # sampled_values: [B, Q, heads, num_points, head_dim]
        return sampled_values

    def forward(self, reference_points, spatial_shape, query, value, key_padding_mask = None):
        """
        - query: [batch, query_len, embed_dim]
        - value: [batch, H~ * W~, embed_dim]
        - spatial_shape: [H~, W~] (height, width of each feature level)
        - reference_points: [batch, query_len, 2]

        - output:       [batch, query_len, embed_dim]
        - attn_weights: [batch, query_len, heads, num_points]
        """
        H, W = spatial_shape
        batch_size = query.size(0)
        # project the input value
        v = self.v_proj(value)
        if key_padding_mask is not None:
            v = v.masked_fill(key_padding_mask[..., None], 0.0)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # [batch, heads, H~ * W~, head_dim]
        # get the attention weights for each query
        attn_weights = self.attention(query) # [batch, query_len, heads * num_points]
        attn_weights = attn_weights.view(batch_size, -1, self.num_heads, self.num_points) # [batch, query_len, heads, num_points]
        attn_weights = F.softmax(attn_weights, dim=-1) # [B, Q, heads, num_points]
        attn_weights = self.dropout(attn_weights)
        # learned offsets for deformable attention
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(batch_size, -1, self.num_heads, self.num_points, 2) # [batch, query_len, heads, num_points, 2]
        # get the sampling points by adding the offsets to the reference points
        offset_normalizer = torch.tensor([W, H], device=query.device, dtype=query.dtype)
        sampling_locations = reference_points[:, :, None, None, :] + sampling_offsets / offset_normalizer
        sampled_values = self._sample_at_points(v, sampling_locations, spatial_shape) # [B, Q, heads, num_points, head_dim]
        # sampled_values: [B, Q, heads, num_points, head_dim]
        # attn_weights: [B, Q, heads, num_points]
        output = sampled_values * attn_weights[..., None] # [B, Q, heads, num_points, head_dim]
        output = output.sum(dim=3) # sum over the sampled points
        output = output.view(batch_size, -1, self.embed_dim) # [B, Q, embed_dim]
        output = self.out_proj(output) # [B, Q, embed_dim]
        return output, attn_weights, sampling_locations
    

class MultiscaleDeformableAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int = 4):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        # linear projections for query, key, and value
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points) # attention weights are learnable
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2) # learnable offsets for deformable attention
        # initialize the parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialisation of parameters:
        - attention_weights: initialized at 0
        - sampling_offsets: initialized at 0
        - v_proj: randomly initialized (xavier uniform)
        - out_proj: randomly initialized (xavier uniform)
        """
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.zeros_(self.sampling_offsets.weight)
        # TODO: implement radial initialization as in the original deformable DETR
        # nn.init.zeros_(self.sampling_offsets.bias)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        ) # we have here 4 attention heads, so it will here define the directions along the y and x axis
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))
        # xavier uniform initialization for v_proj and out_proj
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)


    def _sample_at_points(self, values, sampling_locations, spatial_shapes, attn_weights):
        """
        - values: [batch, heads, sum_l(Hl * Wl), head_dim]
        - sampling_locations: [batch, query_len, heads, num_levels, num_points, 2]
        - spatial_shapes: [num_levels, 2] (height, width)
        - attn_weights: [batch, query_len, heads, num_levels, num_points]

        - output: [B, Q, heads, num_levels, head_dim]

        We sample points from the value feature map at the locations in sampling_locations.
        For each head, we sample num_points points.
        """
        # so we need to do sampling for each level separately
        # we split the values into num_levels parts according to the spatial shapes
        # [batch, heads, sum_l(Hl * Wl), head_dim] -> [batch, sum_l(Hl * Wl), heads, head_dim]
        values = values.transpose(1, 2)
        split_sizes = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist() # [H0~ * W0~, H1~ * W1~, ...]
        # list of num_levels tensors, each [batch, Hl * Wl, heads, head_dim]
        value_list = values.split(split_sizes, dim=1)
        # normalize the sampling locations for the grid_sample in [-1, 1]
        sampling_grids = sampling_locations * 2 - 1

        outputs = []
        # loop over each level
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            value = value_list[level] # [batch, Hl~ * Wl~, heads, head_dim]
            B, _, H, D = value.size()
            _, Q, _, _, _, _ = sampling_locations.size()
            # [batch * heads, head_dim, Hl, Wl]
            value = value.permute(0, 2, 3, 1).contiguous()
            value = value.view(B * self.num_heads, D, height, width)
            # now we need to do the same for the sampling grid
            sampling_grid = sampling_grids[:, :, :, level, :, :] # [batch, query_len, heads, num_points, 2]
            # [batch, query_len, heads, num_points, 2] -> [batch * heads, query_len, num_points, 2]
            grid = sampling_grid.permute(0, 2, 1, 3, 4).contiguous()
            grid = grid.view(B * self.num_heads, -1, self.num_points, 2)
            # now sample the values at the sampling locations using grid_sample
            sampled_value = F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            # sampled_value: [B * heads, head_dim, query_len, num_points]
            # we need to reshape it back to [B, query_len, heads, num_points, head_dim]
            # [B * heads, head_dim, query_len, num_points] -> [B, heads, head_dim, query_len, num_points]
            sampled_value = sampled_value.view(B, self.num_heads, D, Q, self.num_points)
            # [B, heads, head_dim, query_len, num_points] -> [B, query_len, heads, num_points, head_dim]
            sampled_value = sampled_value.permute(0, 3, 1, 4, 2).contiguous()
            # now apply the attention weights
            attn_weight = attn_weights[:, :, :, level, :] # [B, query_len, heads, num_points]
            output = sampled_value * attn_weight[..., None] # [B, Q, heads, num_points, head_dim]
            # [B, Q, heads, num_points, head_dim] -> [B, Q, heads, head_dim]
            output = output.sum(dim=3) # sum over the sampled points
            # now we simply stack the sampled values from different levels together
            outputs.append(output)
        outputs = torch.stack(outputs, dim=3) # [B, Q, heads, num_levels, head_dim]
        return outputs

    def forward(self, query, reference_points, values, spatial_shapes, key_padding_mask = None):
        """
        - query: [batch, query_len, embed_dim]
        - values: [batch, sum_l(Hl * Wl), embed_dim]
        - spatial_shapes: [num_levels, 2] (height, width of each feature level)
        - reference_points: [batch, query_len, num_levels, 2]
        - key_padding_mask: [B, 1, query_len]

        - output:       [batch, query_len, embed_dim]
        - attn_weights: [batch, query_len, heads, num_levels, num_points]

        The values are the concatenation of the feature maps from different levels.
        The `values` tensor is therefore of shape [B, sum_l(Hl~ * Wl~), embed_dim], where Hl~ and Wl~ are the height and width of the feature map at level l.
        For each batch B:
            For each query Q:
                For each head H:
                    For each level L:
                        For each point P:
        """
        batch_size = query.size(0)
        # project the input value
        v = self.v_proj(values) # [B, sum_l(Hl * Wl), embed_dim]
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 3:
                key_padding_mask = key_padding_mask.squeeze(1) # [B, sum_l(Hl, Wl)]
            key_padding_mask = key_padding_mask.to(torch.bool)
            v = v.masked_fill(key_padding_mask[..., None], 0.0)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        # [B, heads, sum_l(Hl~ * Wl~), head_dim]
        v = v.transpose(1, 2)
        # get the attention weights for each query
        attn_weights = self.attention_weights(query) # [batch, query_len, heads * num_levels * num_points]
        attn_weights = attn_weights.view(batch_size, -1, self.num_heads, self.num_levels * self.num_points) # [batch, query_len, heads, num_levels * num_points]
        attn_weights = F.softmax(attn_weights, dim=-1) # [B, Q, heads, num_levels * num_points]
        attn_weights = attn_weights.view(batch_size, -1, self.num_heads, self.num_levels, self.num_points) # [batch, query_len, heads, num_levels, num_points]
        # learned offsets for deformable attention
        sampling_offsets = self.sampling_offsets(query)
        # [batch, query_len, heads, num_levels, num_points, 2]
        sampling_offsets = sampling_offsets.view(batch_size, -1, self.num_heads, self.num_levels, self.num_points, 2)
        # get the sampling points by adding the offsets to the reference points
        spatial_shapes = spatial_shapes.to(device=query.device, dtype=torch.long) # put on the GPU
        offset_normalizer = torch.stack(
            [spatial_shapes[:, 1], spatial_shapes[:, 0]],
            dim=-1,
        ).to(dtype=query.dtype) # [num_levels, 2] (width, height) for normalizing the offsets
        offset_normalizer = offset_normalizer.to(dtype=query.dtype)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        output = self._sample_at_points(v, sampling_locations, spatial_shapes, attn_weights) # [B, Q, heads, num_levels, head_dim]
        # [B, Q, heads, num_levels, head_dim] -> [B, Q, heads, head_dim]
        output = output.sum(dim=3) # sum over the levels
        # [B, Q, heads, head_dim] -> [B, Q, embed_dim]
        output = output.view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output) # [B, Q, embed_dim]
        return output, attn_weights, sampling_locations





class MultiscaleMultireferencesDeformableAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int = 4, num_ref_points_per_query: int = 1):
        """
        This is an extension of Multiscale Deformable Attention proposed in paper Deformable DETR.
        In the original paper, each query element (elements in `query` or length `query_len`) is associated with one reference point.
        In this work, we extend the concept to allow for n-set matching, where n is the number of reference points per query.

        The decoder of DETR while have `query_len` slots for object detection.
        Therefore, `query_len` reference points are normally generated.
        The position of an object is learned as the relative position from the reference point.

        We extend this model to allow for pair (or n-object set) detection inside one query.
        On query then represent an abstract object, and each reference point represent a concrete object.

        Why doing that? In the case I want to apply it, I want to do pair recognition of <button, hole> on clothes.
        Buttons are small targets, and holes are small or invisible targets: while buttons are objects, holes are more like key points.
        The idea is to have an abstract object representing this pair <button, hole>, and have 2 reference points that will represent the button and the hole.
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.num_ref_points_per_query = num_ref_points_per_query # let's call it RpQ
        # linear projections for query, key, and value
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points * num_ref_points_per_query) # attention weights are learnable
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * num_ref_points_per_query * 2) # learnable offsets for deformable attention
        # initialize the parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialisation of parameters:
        - attention_weights: initialized at 0
        - sampling_offsets: initialized at 0
        - v_proj: randomly initialized (xavier uniform)
        - out_proj: randomly initialized (xavier uniform)
        """
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.zeros_(self.sampling_offsets.weight)
        # TODO: do proper initialization
        """
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        ) # we have here 4 attention heads, so it will here define the directions along the y and x axis
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))
        """
        nn.init.uniform_(self.sampling_offsets.bias)
        # xavier uniform initialization for v_proj and out_proj
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)


    def _sample_at_points(self, values, sampling_locations, spatial_shapes, attn_weights):
        """
        - values: [batch, heads, sum_l(Hl * Wl), head_dim]
        - sampling_locations: [batch, query_len, heads, num_levels, num_points, RpQ, 2]
        - spatial_shapes: [num_levels, 2] (height, width)
        - attn_weights: [batch, query_len, heads, num_levels, num_points, RpQ]

        - output: [B, Q, heads, num_levels, RpQ, head_dim]

        We sample points from the value feature map at the locations in sampling_locations.
        For each head, we sample `num_points` points.
        """
        # so we need to do sampling for each level separately
        # we split the values into num_levels parts according to the spatial shapes
        # [batch, heads, sum_l(Hl * Wl), head_dim] -> [batch, sum_l(Hl * Wl), heads, head_dim]
        values = values.transpose(1, 2)
        split_sizes = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist() # [H0~ * W0~, H1~ * W1~, ...]
        # list of num_levels tensors, each [batch, Hl * Wl, heads, head_dim]
        value_list = values.split(split_sizes, dim=1)
        # normalize the sampling locations for the grid_sample in [-1, 1]
        # [batch, query_len, heads, num_levels, num_points, RpQ, 2]
        sampling_grids = sampling_locations * 2 - 1

        outputs = []
        # loop over each level
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            value = value_list[level] # [batch, Hl~ * Wl~, heads, head_dim]
            B, _, H, D = value.size()
            _, Q, _, _, _, _, _ = sampling_locations.size()
            # IMPORTANT (value): order of flatten: B, RqP, heads 
            # [batch, Hl~ * Wl~, heads, head_dim]
            # -> [batch, 1, Hl~ * Wl~, heads, head_dim]
            # -> [batch, RpQ, Hl~ * Wl~, heads, head_dim]
            # -> [batch, RpQ, heads, head_dim, Hl~ * Wl~]
            # [batch * heads * RpQ, head_dim, Hl, Wl]
            value = value[:, None, :, :, :].expand(B, self.num_ref_points_per_query, height * width, H, D)
            value = value.permute(0, 1, 3, 4, 2).contiguous()
            value = value.view(B * self.num_heads * self.num_ref_points_per_query, D, height, width)
            # now we need to do the same for the sampling grid
            sampling_grid = sampling_grids[:, :, :, level, :, :] # [batch, query_len, heads, num_points, RpQ, 2]
            # [batch, query_len, heads, num_points, RpQ, 2]
            # -> [batch, RpQ, heads, query_len, num_points, 2]
            # -> [batch * RpQ * heads, query_len, num_points, 2]
            grid = sampling_grid.permute(0, 4, 2, 1, 3, 5).contiguous()
            grid = grid.view(B * self.num_heads * self.num_ref_points_per_query, -1, self.num_points, 2)
            # now sample the values at the sampling locations using grid_sample
            sampled_value = F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            # sampled_value: [B * heads * RpQ, head_dim, query_len, num_points]
            # we need to reshape it back to [B, query_len, heads, num_points, RpQ, head_dim]
            # IMPORTANT (sampled_value): order of deflatten (same as `value`): B, heads, RqP
            # [B * heads * RpQ, head_dim, query_len, num_points] -> [B, RpQ, heads, head_dim, query_len, num_points]
            sampled_value = sampled_value.view(B, self.num_ref_points_per_query, self.num_heads, D, Q, self.num_points)
            # [B, RpQ, heads, head_dim, query_len, num_points] -> [B, query_len, heads, num_points, RpQ, head_dim]
            sampled_value = sampled_value.permute(0, 4, 2, 5, 1, 3).contiguous()
            # now apply the attention weights
            attn_weight = attn_weights[:, :, :, level, :, :] # [B, query_len, heads, num_points, RpQ]
            output = sampled_value * attn_weight[..., None] # [B, Q, heads, num_points, RpQ, head_dim]
            # [B, Q, heads, num_points, RpQ, head_dim] -> [B, Q, heads, RpQ, head_dim]
            output = output.sum(dim=3) # sum over the sampled points
            # now we simply stack the sampled values from different levels together
            outputs.append(output)
        outputs = torch.stack(outputs, dim=3) # [B, Q, heads, num_levels, RpQ, head_dim]
        return outputs

    def forward(self, query, reference_points, values, spatial_shapes, key_padding_mask = None):
        """
        - query: [batch, query_len, embed_dim]
        - values: [batch, sum_l(Hl * Wl), embed_dim]
        - spatial_shapes: [num_levels, 2] (height, width of each feature level)
        - reference_points: [batch, query_len, num_levels, RpQ, 2]

        - output:       [batch, query_len, embed_dim]
        - attn_weights: [batch, query_len, heads, num_levels, num_points]

        The values are the concatenation of the feature maps from different levels.
        The `values` tensor is therefore of shape [B, sum_l(Hl~ * Wl~), embed_dim], where Hl~ and Wl~ are the height and width of the feature map at level l.
        For each batch B:
            For each query Q:
                For each head H:
                    For each level L:
                        For each point P:
        """
        batch_size = query.size(0)
        # project the input value
        v = self.v_proj(values) # [B, sum_l(Hl * Wl), embed_dim]
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 3:
                key_padding_mask = key_padding_mask.squeeze(1)
            key_padding_mask = key_padding_mask.to(torch.bool)
            v = v.masked_fill(key_padding_mask[..., None], 0.0)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        # [B, heads, sum_l(Hl~ * Wl~), head_dim]
        v = v.transpose(1, 2)
        # get the attention weights for each query
        # [batch, query_len, heads * num_levels * num_points * RpQ]
        attn_weights = self.attention_weights(query)
        # [batch, query_len, heads * num_levels * num_points * RpQ] -> [batch, query_len, heads, num_levels, num_points, RpQ]
        attn_weights = attn_weights.view(batch_size, -1, self.num_heads, self.num_levels, self.num_points, self.num_ref_points_per_query)
        # [batch, query_len, heads, num_levels, num_points, RpQ] -> [batch, query_len, heads, RpQ, num_levels, num_points]
        attn_weights = attn_weights.permute(0, 1, 2, 5, 3, 4).contiguous()
        # [batch, query_len, heads, RpQ, num_levels, num_points] -> [batch, query_len, heads, RpQ, num_levels * num_points]
        attn_weights = attn_weights.view(batch_size, -1, self.num_heads, self.num_ref_points_per_query, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1) # [B, Q, heads, RpQ, num_levels * num_points]
        # [B, Q, heads, RpQ, num_levels * num_points] -> [batch, query_len, heads, RpQ, num_levels, num_points]
        attn_weights = attn_weights.view(batch_size, -1, self.num_heads, self.num_ref_points_per_query, self.num_levels, self.num_points)
        # [batch, query_len, heads, RpQ, num_levels, num_points] -> [batch, query_len, heads, num_levels, num_points, RpQ]
        attn_weights = attn_weights.permute(0, 1, 2, 4, 5, 3).contiguous()
        # learned offsets for deformable attention
        sampling_offsets = self.sampling_offsets(query)
        # [batch, query_len, heads, num_levels, num_points, RpQ, 2]
        sampling_offsets = sampling_offsets.view(batch_size, -1, self.num_heads, self.num_levels, self.num_points, self.num_ref_points_per_query, 2)
        # get the sampling points by adding the offsets to the reference points
        spatial_shapes = spatial_shapes.to(device=query.device, dtype=torch.long) # put on the GPU
        offset_normalizer = torch.stack(
            [spatial_shapes[:, 0], spatial_shapes[:, 1]],
            dim=-1,
        ).to(dtype=query.dtype) # [num_levels, 2] (width, height) for normalizing the offsets
        offset_normalizer = offset_normalizer.to(dtype=query.dtype)
        # [batch, query_len, num_levels, RpQ, 2]
        sampling_locations = reference_points[:, :, None, :, None, :, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, None, :]
        output = self._sample_at_points(v, sampling_locations, spatial_shapes, attn_weights) # [B, Q, heads, num_levels, RpQ, head_dim]
        # [B, Q, heads, num_levels, RpQ, head_dim] -> [B, Q, heads, RpQ, head_dim]
        output = output.sum(dim=3) # sum over the levels
        
        # ==================================================================
        # ========================== IMPORTANT =============================
        # ==================================================================
        # ==== Here we can do a few things:                             ====
        # ==== - take the mean over RpQ, but we lose information        ====
        # ==== - simply return as it so that we don't lose information  ====
        # ==================================================================
        # Let's implement the second option.
        # [B, Q, heads, RpQ, head_dim] -> [B, Q, RpQ, heads, head_dim]
        output = output.permute(0, 1, 3, 2, 4).contiguous()
        # [B, Q, RpQ, heads, head_dim] -> [B, Q, RpQ, embed_dim]
        output = output.view(batch_size, -1, self.num_ref_points_per_query, self.embed_dim)
        output = self.out_proj(output) # [B, Q, RpQ, embed_dim]
        return output, attn_weights, sampling_locations
