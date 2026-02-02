import os
import numpy as np
from tokenizers import Tokenizer

# 将数据集的token id转换为bin
def generate_bin():
    # 准备目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 2. 加载分词器 ---
    print(f"Loading tokenizer: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    # --- 3. 读取文本 ---
    print(f"Reading text: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    # --- 4. 编码 (最关键一步) ---
    # 把 "罗辑" 变成 [5000] 这样的数字
    print("Encoding text to IDs...")
    encoded = tokenizer.encode(text)
    ids = encoded.ids
    
    # 添加结束符 (End of Text)，有助于模型学习何时停止
    # 这里的 0 必须对应你 json 文件里 <|endoftext|> 的 ID
    # 这样切分后训练集后是没有id0的。
    # 影响：对于《三体》单行本训练，影响为零。模型会把训练集当成一个无限循环的故事来读。
    ids.append(0) 
    
    total_tokens = len(ids)
    print(f"Total tokens: {total_tokens}")

    # --- 5. 转换为 uint16 (核心优化) ---
    # 因为你的词表约 20000 < 65535，所以用 uint16 (2字节) 存。
    # 这比 Python 默认的 int (通常8字节) 或 int32 (4字节) 极其节省空间。
    data = np.array(ids, dtype=np.uint32)

    # --- 6. 划分训练集和验证集 划分数据集 (80% / 10% / 10%) ---
    # 注意这里是顺序划分
    # 对于小说、代码这种强逻辑、强时序的数据，必须像切蛋糕一样一刀切。
    # 只有对于互不相关的独立样本（比如 10000 条独立的微博评论做情感分析），才使用随机划分。
    split_idx1 = int(total_tokens * 0.8)
    split_idx2 = int(total_tokens * 0.9)
    # 训练集: 0% -> 80%
    train_data = data[:split_idx1]
    # 验证集: 80% -> 90%
    val_data = data[split_idx1:split_idx2]
    # 测试集: 90% -> 100%
    test_data = data[split_idx2:]

    # --- 7. 写入硬盘 ---
    # .tofile() 会保存纯二进制数据，不带任何文件头信息
    train_path = os.path.join(OUTPUT_DIR, "train.bin")
    val_path = os.path.join(OUTPUT_DIR, "val.bin")
    test_path = os.path.join(OUTPUT_DIR, "test.bin")
    
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    test_data.tofile(test_path)

    print("-" * 30)
    print(f"🎉 生成完成！")
    print(f"训练集: {train_path} ({len(train_data)} tokens)")
    print(f"验证集: {val_path} ({len(val_data)} tokens)")
    print(f"验证集: {test_path} ({len(test_data)} tokens)")
    
    # 计算给 C 语言用的词表大小 (对齐到 64 的倍数)
    padded_vocab = ((vocab_size + 63) // 64) * 64
    vocab_size_path = os.path.join(OUTPUT_DIR, "vocab_log.txt")
    with open(vocab_size_path, 'w', encoding='utf-8') as f:
        f.write(f"Total tokens={total_tokens}\n")
        f.write(f"训练集: {train_path} ({len(train_data)} tokens)\n")
        f.write(f"验证集: {val_path} ({len(val_data)} tokens)\n")
        f.write(f"验证集: {test_path} ({len(test_data)} tokens)\n")
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"padded_vocab={padded_vocab}\n")
    print(f"\n💡 下一步运行 llm.c 时，请使用参数: -v {padded_vocab}")
    
import json
import struct
import sys
import os
# 将词表转换为bin

def convert_tokenizer(json_path, bin_path):
    print(f"正在加载 Tokenizer 文件: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ---------------------------------------------------------
    # 1. 提取词表 (Vocab Extraction)
    # ---------------------------------------------------------
    vocab_map = {}
    
    # 情况 A: HuggingFace 标准 tokenizer.json 结构
    if "model" in data and "vocab" in data["model"]:
        # print("识别为 HuggingFace 'tokenizer.json' 格式")
        vocab_map = data["model"]["vocab"]
    # 情况 B: 简单的键值对 vocab.json
    elif isinstance(data, dict):
        first_val = next(iter(data.values()))
        if isinstance(first_val, int):
            # print("识别为简单字典格式 {'token': id}")
            vocab_map = data
        else:
            print("警告: 无法识别的 JSON 结构，尝试在根目录寻找 'vocab' 字段")
            if "vocab" in data:
                vocab_map = data["vocab"]
    
    if not vocab_map:
        raise ValueError("无法在 JSON 中找到词表 (vocab) 数据！")

    # ---------------------------------------------------------
    # 2. 排序与对齐 (Sorting & Alignment) --- 【核心修改点】
    # ---------------------------------------------------------
    # 找出实际最大的 ID
    max_id = max(vocab_map.values())
    original_vocab_size = max_id + 1
    
    # 【关键步骤】强制向上对齐到 64 的倍数
    # 这会把 20000 变成 20032，从而包含模型可能预测出的 "越界" ID
    vocab_size = ((original_vocab_size + 63) // 64) * 64
    
    print(f"原始词表大小: {original_vocab_size}")
    print(f"对齐后大小 (Padding): {vocab_size} (这将是 C 程序读取的大小)")
    
    # 初始化列表，长度为对齐后的大小
    token_list = [None] * vocab_size
    
    # 填入真实的词
    for token_str, token_id in vocab_map.items():
        if 0 <= token_id < vocab_size:
            token_list[token_id] = token_str
        else:
            print(f"警告: 跳过异常 ID {token_id}: {token_str}")

    # ---------------------------------------------------------
    # 3. 填补空洞 (Gap Filling)
    # ---------------------------------------------------------
    fill_count = 0
    # 遍历整个【对齐后】的大小
    for i in range(vocab_size):
        if token_list[i] is None:
            # 【关键步骤】用 <pad_ID> 填补空位
            # 这样当 C 程序读到 ID 20001 时，会打印 "<pad_20001>" 而不是崩溃
            token_list[i] = f"<pad_{i}>"
            fill_count += 1
    
    if fill_count > 0:
        print(f"已自动填充 {fill_count} 个空洞 (含 Padding)")

    # ---------------------------------------------------------
    # 4. 写入二进制文件 (Binary Writing)
    # ---------------------------------------------------------
    print(f"正在写入二进制文件: {bin_path}")
    
    with open(bin_path, 'wb') as f:
        # --- Header (1024 bytes) ---
        f.write(struct.pack('<I', 20260123))   # Magic
        f.write(struct.pack('<I', 1))          # Version
        f.write(struct.pack('<I', vocab_size)) # 【注意】写入的是对齐后的大小(20032)
        f.write(b'LJJ\x00')                    # Creator
        
        # 填充 Header 剩余部分
        for _ in range(252):
            f.write(struct.pack('<I', 0))

        # --- Body (Token Data) ---
        for i, token_str in enumerate(token_list):
            # 处理特殊字符
            token_str = token_str.replace('Ġ', ' ') 
            token_bytes = token_str.encode('utf-8')
            length = len(token_bytes)
            
            # C 代码限制长度必须是 0-255
            if length > 255:
                token_bytes = token_bytes[:255]
                length = 255
            elif length == 0:
                token_bytes = b'\0'
                length = 1 

            f.write(struct.pack('<B', length))
            f.write(token_bytes)

    print("✅ 词表转换完成！")

if __name__ == "__main__":
    # --- 1. 配置路径 (请根据你的实际情况修改) ---
    TOKENIZER_PATH = "output_tokenizer/threebody_tokenizer.json" # 你刚才生成的json
    INPUT_FILE = "datasets/三体全集"                         # 你的小说txt
    OUTPUT_DIR = "output_tokenizer"                         # 准备存放bin文件的目录
    # 将数据集token id 转换为bin
    generate_bin()
    
    # 将词表转换为bin
    output_bin = os.path.join(OUTPUT_DIR, "threebody_tokenizer.bin") # 输出的 BIN 文件路径

    if not os.path.exists(TOKENIZER_PATH):
        print(f"错误: 找不到文件 {TOKENIZER_PATH}")
    else:
        try:
            convert_tokenizer(TOKENIZER_PATH, output_bin)
        except Exception as e:
            print(f"转换失败: {e}")