from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import os
import glob
import re

# --- 配置 ---
BASE_MODEL_NAME = "threebody_tokenizer"
OUTPUT_DIR = "output_tokenizer"
DATASET_DIR = "./datasets"
TEMP_CORPUS_FILE = "temp_merged_corpus.txt" # 临时合并文件

def clean_text(text):
    """
    数据清洗函数：只保留有用的文本
    """
    # 1. 替换全角空格为半角空格 (很多中文小说会有 \u3000)
    text = text.replace('\u3000', ' ')
    
    # 2. 去除不可见字符 (除了换行符 \n 和 制表符 \t)
    # 这一步是为了防止 weird control characters 进入词表
    text = "".join(ch for ch in text if ch.isprintable() or ch in ['\n', '\t'])

    # 3. 把连续的多个空格变成一个空格 (可选，看你是否在意缩进)
    text = re.sub(r'\s+', ' ', text) 
    
    # 4. 去除连续的空行 (保留段落结构，但去除大段空白)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text

def merge_and_clean_files(source_dir, output_file):
    """
    读取目录下所有txt，清洗后合并到一个临时文件
    """
    # 找到所有 txt 文件
    files = glob.glob(os.path.join(source_dir, "*.txt"))
    if not files:
        raise ValueError(f"在 {source_dir} 下没有找到 .txt 文件！")
    
    print(f"📚 发现 {len(files)} 个文件，准备合并清洗...")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in files:
            print(f"  -> 处理: {os.path.basename(file_path)}")
            try:
                # errors='ignore' 防止因为某个字编码错误导致整个脚本崩溃
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    cleaned_content = clean_text(content)
                    outfile.write(cleaned_content)
                    # 每个文件之间加个换行，防止前一本书的结尾和后一本书的开头连在一起
                    outfile.write("\n") 
            except Exception as e:
                print(f"⚠️ 跳过文件 {file_path}: {e}")
    
    print(f"✅ 合并完成，生成临时语料: {output_file}")
    return output_file

def train_tokenizer(corpus_file):
    # 1. 初始化 Tokenizer (BPE)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. 预处理 (ByteLevel)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # 3. 解码器
    tokenizer.decoder = decoders.BPEDecoder()

    # 4. 训练器配置
    # 注意：如果语料变大了（刘慈欣全集），20000 依然是合理的，
    # 但如果语料极其巨大（GB级别），可能需要考虑 30000-50000。
    trainer = trainers.BpeTrainer(
        vocab_size=20000, 
        min_frequency=2,
        special_tokens=[
            "[UNK]",
            "<|endoftext|>", "<|padding|>", 
            "<|im_start|>", "<|im_end|>", 
            "<|system|>", "<|user|>", "<|assistant|>",
            "<|thought|>", "<|/thought|>"
        ],
        show_progress=True
    )

    # 5. 开始训练
    print(f"🚀 开始训练 Tokenizer，读取合并语料: {corpus_file} ...")
    # 注意：这里直接传入文件路径列表
    tokenizer.train([corpus_file], trainer=trainer)

    # 6. 保存
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    save_path = os.path.join(OUTPUT_DIR, "threebody_tokenizer.json")
    tokenizer.save(save_path)
    print(f"💾 训练完成！已保存至: {save_path}")
    return tokenizer

if __name__ == "__main__":
    model_path = os.path.join(OUTPUT_DIR, "threebody_tokenizer.json")
    
    # 强制重新训练的开关（如果你想更新词表，设为 True）
    FORCE_RETRAIN = False 

    if not os.path.exists(model_path) or FORCE_RETRAIN:
        # 1. 清洗并合并数据
        merge_and_clean_files(DATASET_DIR, TEMP_CORPUS_FILE)
        
        # 2. 训练
        tokenizer = train_tokenizer(TEMP_CORPUS_FILE)
        
        # 3. 删除临时文件 (可选)
        if os.path.exists(TEMP_CORPUS_FILE):
            os.remove(TEMP_CORPUS_FILE)
            print("🗑️ 已删除临时合并文件")
    else:
        print(f"🔍 发现已存在的模型: {model_path}")
        tokenizer = Tokenizer.from_file(model_path)
        print("✅ 加载成功！")

    # --- 测试环节 ---
    print("\n" + "="*30)
    # 测试一些小说里常见的词，看看它们是一个 Token 还是被拆分了
    test_sentences = [
        "不要回答！不要回答！不要回答！",
        "罗辑直接向三体世界发出了威慑。",
        "给岁月以文明，而不是给文明以岁月。", # 黑暗森林名言
        "弱小和无知不是生存的障碍，傲慢才是。", # 死神永生名言
        "章北海微微一笑。",
        "这是刘慈欣的科幻小说全集。"
    ]

    for text in test_sentences:
        encoded = tokenizer.encode(text)
        print(f"\n原文: {text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"IDs:    {encoded.ids}")