import torch
import torch.nn.functional as F


def convert_mha_to_gqa(mha_ckpt_path, gqa_ckpt_path, hidden_size=4096, num_attention_heads=64, num_key_value_heads=8):
    mha_state = torch.load(mha_ckpt_path)
    gqa_state = {}

    assert num_attention_heads % num_key_value_heads == 0

    kv_group_size = int(num_attention_heads / num_key_value_heads)

    for k, v in mha_state.items():
        if 'attention.query_key_value.weight' in k:
            # mha: [12288, 4096] -> gqa: [5120, 4096]
            q_w, kv_w = v.split([hidden_size, 2 * hidden_size])
            kv_w = kv_w.view(2, num_attention_heads, -1)
            kv_pooled = F.avg_pool2d(kv_w, (kv_group_size, 1)).view(-1, hidden_size)
            gqa_qkv_w = torch.cat((q_w, kv_pooled), 0)
            v = gqa_qkv_w
        elif 'attention.query_key_value.bias' in k:
            # mha: [12288] -> gqa: [5120]
            q_b, kv_b = v.split([hidden_size, 2 * hidden_size])
            kv_b = kv_b.view(2, num_attention_heads, -1)
            kv_pooled = F.avg_pool2d(kv_b, (kv_group_size, 1)).view(-1)
            gqa_qkv_b = torch.cat((q_b, kv_pooled), 0)
            v = gqa_qkv_b
        if k.startswith("transformer.transformer."):
            k = k[12:]  # remove the starting prefix transformer.
        gqa_state[k] = v
    torch.save(gqa_state, gqa_ckpt_path)
