import os
import glob
import json
import struct
import re
import numpy as np
import logging  # 引入 logging 模块
from tokenizers import Tokenizer

# ================= 配置区域 =================
TOKENIZER_JSON_PATH = "output_tokenizer/threebody_tokenizer.json"
DATASET_DIR = "datasets" 
OUTPUT_DIR = "output_tokenizer"
LOG_FILE = os.path.join(OUTPUT_DIR, "data_processing.log") # Log 文件路径

# ⚠️ 必须检查你的 json 文件，确认 <|endoftext|> 的 ID 是多少
EOT_TOKEN_ID = 1  
# ===========================================

def clean_text(text):
    """
    文本清洗函数 (保持不变)
    """
    text = text.replace('\u3000', ' ')
    text = "".join(ch for ch in text if ch.isprintable() or ch in ['\n', '\t'])
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()
    return text

def generate_dataset_bin():
    """
    生成数据集 bin 文件，并记录日志
    """
    logging.info(">>> 阶段 1/2: 开始生成数据集 bin 文件")
    print(f"\n[1/2] 正在生成数据集 bin 文件 (32-bit mode)...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 加载分词器
    if not os.path.exists(TOKENIZER_JSON_PATH):
        error_msg = f"找不到 Tokenizer 文件: {TOKENIZER_JSON_PATH}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    print(f"Loading tokenizer: {TOKENIZER_JSON_PATH}")
    tokenizer = Tokenizer.from_file(TOKENIZER_JSON_PATH)
    vocab_size = tokenizer.get_vocab_size()
    
    msg = f"Tokenizer 加载成功, Vocab size: {vocab_size}"
    print(msg)
    logging.info(msg)

    # 2. 扫描文件
    txt_files = glob.glob(os.path.join(DATASET_DIR, "*.txt"))
    if not txt_files:
        error_msg = f"在 {DATASET_DIR} 下没有找到 .txt 文件！"
        logging.error(error_msg)
        raise ValueError(error_msg)
    txt_files.sort()

    master_train = []
    master_val = []
    master_test = []

    print(f"发现 {len(txt_files)} 个文件，开始逐个清洗并划分 (8:1:1)...")
    logging.info(f"发现 {len(txt_files)} 个源文件: {[os.path.basename(f) for f in txt_files]}")

    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        print(f"  -> 处理: {file_name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            # --- 步骤 A: 清洗 ---
            cleaned_text = clean_text(raw_text)
            
            # --- 步骤 B: 编码 ---
            encoded = tokenizer.encode(cleaned_text)
            ids = encoded.ids
            
            # 在每本书末尾加 EOT
            ids.append(EOT_TOKEN_ID)
            
            # --- 步骤 C: 按文件划分 (8:1:1) ---
            n_tokens = len(ids)
            if n_tokens < 10: 
                msg = f"警告: {file_name} 内容太少 ({n_tokens} tokens)，跳过划分，全部放入 Train"
                print(f"     ⚠️ {msg}")
                logging.warning(msg)
                master_train.extend(ids)
                continue

            split_80 = int(n_tokens * 0.8)
            split_90 = int(n_tokens * 0.9)
            
            chunk_train = ids[:split_80]
            chunk_val   = ids[split_80:split_90]
            chunk_test  = ids[split_90:]
            
            master_train.extend(chunk_train)
            master_val.extend(chunk_val)
            master_test.extend(chunk_test)
            
            log_msg = f"文件 {file_name} 处理完毕: Train={len(chunk_train)}, Val={len(chunk_val)}, Test={len(chunk_test)} (Tokens)"
            print(f"     ✅ Train: {len(chunk_train)}, Val: {len(chunk_val)}, Test: {len(chunk_test)}")
            logging.info(log_msg)

        except Exception as e:
            msg = f"处理文件 {file_name} 时发生错误: {e}"
            print(f"     ❌ {msg}")
            logging.error(msg)

    # 3. 转换为 numpy uint32
    print("\n正在转换为 np.uint32...")
    train_data = np.array(master_train, dtype=np.uint32)
    val_data   = np.array(master_val,   dtype=np.uint32)
    test_data  = np.array(master_test,  dtype=np.uint32)

    # 4. 写入文件
    print("正在写入硬盘...")
    train_path = os.path.join(OUTPUT_DIR, "train.bin")
    val_path   = os.path.join(OUTPUT_DIR, "val.bin")
    test_path  = os.path.join(OUTPUT_DIR, "test.bin")
    
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    test_data.tofile(test_path)

    # 5. 生成统计日志
    print("-" * 30)
    print(f"🎉 数据集生成完毕！")
    print(f"  Train: {len(train_data)} tokens -> {train_path}")
    print(f"  Val  : {len(val_data)}   tokens -> {val_path}")
    print(f"  Test : {len(test_data)}  tokens -> {test_path}")

    padded_vocab = ((vocab_size + 63) // 64) * 64
    
    # 写入 info txt (供 C 读取)
    info_path = os.path.join(OUTPUT_DIR, "dataset_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Vocab Size: {vocab_size}\n")
        f.write(f"Padded Vocab: {padded_vocab}\n")
        f.write(f"Train Tokens: {len(train_data)}\n")
        f.write(f"Val Tokens: {len(val_data)}\n")
        f.write(f"Test Tokens: {len(test_data)}\n")
    
    # 写入 Log
    logging.info("-" * 20)
    logging.info("数据集汇总统计:")
    logging.info(f"Total Train Tokens: {len(train_data)}")
    logging.info(f"Total Val Tokens  : {len(val_data)}")
    logging.info(f"Total Test Tokens : {len(test_data)}")
    logging.info(f"Padded Vocab Size : {padded_vocab} (C语言参数 -v)")
    logging.info(f"Output files: {train_path}, {val_path}, {test_path}")
    
    print(f"💡 C 语言运行参数: -v {padded_vocab}")

def convert_tokenizer_bin():
    """
    将 Tokenizer 词表转换为 C 语言可读的二进制格式 (保持不变，增加日志)
    """
    logging.info(">>> 阶段 2/2: 开始转换 Tokenizer bin")
    print(f"\n[2/2] 正在转换 Tokenizer 词表为二进制...")
    json_path = TOKENIZER_JSON_PATH
    bin_path = os.path.join(OUTPUT_DIR, "threebody_tokenizer.bin")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab_map = {}
    if "model" in data and "vocab" in data["model"]:
        vocab_map = data["model"]["vocab"]
    elif isinstance(data, dict):
        if "vocab" in data: vocab_map = data["vocab"]
        else: vocab_map = data
            
    if not vocab_map: 
        logging.error("JSON 中未找到 vocab 数据")
        raise ValueError("JSON 中未找到 vocab 数据")

    # 对齐
    max_id = max(vocab_map.values()) if vocab_map else 0
    padded_size = ((max_id + 1 + 63) // 64) * 64
    print(f"Vocab Padded to: {padded_size}")
    logging.info(f"Original Max ID: {max_id}, Padded Size: {padded_size}")

    # 填充空洞
    token_list = [None] * padded_size
    for token, idx in vocab_map.items():
        if 0 <= idx < padded_size:
            token_list[idx] = token
            
    for i in range(padded_size):
        if token_list[i] is None:
            token_list[i] = f"<pad_{i}>"

    # 写入二进制
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<I', 20260123)) 
        f.write(struct.pack('<I', 1)) 
        f.write(struct.pack('<I', padded_size)) 
        f.write(b'LJJ_GPT')
        f.write(b'\0' * (1024 - 12 - 7)) 
        
        logging.info("tokenizer_magic_number: 20260123")

        for token in token_list:
            b = token.encode('utf-8')
            length = len(b)
            if length > 255: length = 255
            if length == 0: length = 1; b = b'\0'
            f.write(struct.pack('<B', length))
            f.write(b[:length])

    msg = f"✅ Tokenizer bin 生成完毕: {bin_path}"
    print(msg)
    logging.info(msg)

if __name__ == "__main__":
    # --- 0. 配置日志 ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w' # 每次运行覆盖旧日志，想追加请改为 'a'
    )
    
    # 记录当前运行配置
    logging.info("========================================")
    logging.info("开始执行数据预处理脚本")
    logging.info(f"TOKENIZER_JSON: {TOKENIZER_JSON_PATH}")
    logging.info(f"DATASET_DIR   : {DATASET_DIR}")
    logging.info(f"OUTPUT_DIR    : {OUTPUT_DIR}")
    logging.info(f"EOT_TOKEN_ID  : {EOT_TOKEN_ID}")
    logging.info("========================================")

    # --- 1. 执行任务 ---
    try:
        generate_dataset_bin()
        convert_tokenizer_bin()
        logging.info("所有任务执行成功！")
    except Exception as e:
        logging.critical(f"脚本执行过程中发生致命错误: {e}", exc_info=True)
        print(f"\n❌ 发生错误，请查看日志文件: {LOG_FILE}")