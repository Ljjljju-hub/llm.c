import os
import math
import struct
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from tokenizers import Tokenizer

# ================= 配置区域 =================
# 分词器路径 (用于自动获取 vocab_size)
TOKENIZER_JSON = "output_tokenizer/threebody_tokenizer.json" 

# 模型输出路径
OUTPUT_MODEL_BIN = "output_pre_model/gpt2_init.bin"

# 模型架构参数 (针对《三体》的小型模型)
# 建议: L=6, H=8, E=512, Context=512
CONF_N_LAYER = 6
CONF_N_HEAD = 8
CONF_N_EMBD = 512
CONF_BLOCK_SIZE = 512
# ===========================================

# --- 1. GPT-2 模型定义 (复刻 train_gpt2.py 的结构) ---
class NewGELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权重绑定 (Weight Tying): Embedding 和 Output Layer 共享权重
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化参数 (重要！)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

# --- 2. 导出函数 (适配 llm.c 格式) ---
def write_fp32(tensor, file):
    file.write(tensor.detach().cpu().numpy().astype("float32").tobytes())

def write_tensors(model_tensors, L, file):
    # 按照 C 代码读取的顺序写入
    write_fp32(model_tensors["transformer.wte.weight"], file)
    write_fp32(model_tensors["transformer.wpe.weight"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L): 
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fp32(model_tensors["transformer.ln_f.weight"], file)
    write_fp32(model_tensors["transformer.ln_f.bias"], file)

def write_model(model, filename):
    # Header 格式: [Magic, Version, B, V, L, H, E] + Padding
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326 # Magic number (llm.c 要求的)
    header[1] = 1        # Version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    
    print(f"写入 Header: T={header[2]}, V={header[3]}, L={header[4]}, H={header[5]}, C={header[6]}")

    params = {name: param.cpu() for name, param in model.named_parameters()}
    
    # ===【新增代码开始】===
    # 获取文件所在的目录路径 (例如 "output_pre_model")
    folder_path = os.path.dirname(filename)
    # 如果目录路径不为空且不存在，则创建它
    if folder_path and not os.path.exists(folder_path):
        print(f"正在创建目录: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)
    # ===【新增代码结束】===
    
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        write_tensors(params, model.config.n_layer, file)
    print(f"✅ 模型已保存至: {filename}")

# --- 3. 主程序 ---
def main():
    # 1. 获取准确的 Vocab Size
    if os.path.exists(TOKENIZER_JSON):
        print(f"正在加载分词器配置: {TOKENIZER_JSON}")
        tokenizer = Tokenizer.from_file(TOKENIZER_JSON)
        # 获取基础词表大小
        base_vocab_size = tokenizer.get_vocab_size()
        # 对齐到 64 的倍数 (Padding) - 对 C/CUDA 性能至关重要
        padded_vocab_size = ((base_vocab_size + 63) // 64) * 64
        print(f"检测到词表大小: {base_vocab_size} -> 对齐后: {padded_vocab_size}")
    else:
        print(f"⚠️ 警告: 找不到 {TOKENIZER_JSON}，使用默认词表大小 50257")
        padded_vocab_size = 50257

    # 2. 配置模型
    config = GPTConfig(
        block_size = CONF_BLOCK_SIZE,
        vocab_size = padded_vocab_size, # 使用对齐后的大小
        n_layer = CONF_N_LAYER,
        n_head = CONF_N_HEAD,
        n_embd = CONF_N_EMBD
    )

    # 3. 初始化模型 (Random Initialization)
    print("正在初始化随机权重...")
    model = GPT(config)
    
    # 4. 打印参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M")

    # 5. 保存为 .bin
    write_model(model, OUTPUT_MODEL_BIN)
    
    print("\n" + "="*40)
    print("🚀 准备工作完成！")
    print(f"1. 初始权重文件: {OUTPUT_MODEL_BIN}")
    print(f"2. 参数设置: -v {padded_vocab_size} (运行 llm.c 时请务必使用此参数)")
    print("="*40)
    
    folder_path = os.path.dirname(OUTPUT_MODEL_BIN)
    file_name = os.path.join(folder_path, "log.txt")
    with open(file_name, "w") as f:
        f.write(f"1. 初始权重文件: {OUTPUT_MODEL_BIN}\n")
        f.write(f"2. 模型的词表大小：参数设置: -v {padded_vocab_size} (运行 llm.c 时请务必使用此参数)\n")

if __name__ == "__main__":
    main()