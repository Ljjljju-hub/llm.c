from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import os
import re
from datetime import datetime

# 定义基础文件名
BASE_MODEL_NAME = "threebody_tokenizer"
# 保存文件夹
output_dir = "output_tokenizer"

def train_threebody_tokenizer(data_file):
    # 1. 初始化 Tokenizer
    # 使用 BPE 模型（GPT-2/3/4, LLaMA 同款核心算法）
    tokenizer = Tokenizer(models.BPE())

    # 2. 预处理 (Pre-tokenization)
    # ByteLevel 极其重要：它将字符转化为字节。
    # 这意味着任何 Unicode 字符（包括生僻汉字）都能被处理，不会出现 [UNK]
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 3. 解码器 (Decoder)
    # 用于将 ID 转回文本时，把字节还原成字符
    tokenizer.decoder = decoders.ByteLevel()

    # 4. 设置训练器 (Trainer)
    # vocab_size: 词表大小。
    # 《三体》全文约 80-90万字。
    # 常见的中文 LLM 词表在 3万-10万之间。
    # 对于纯《三体》语料，设为 10,000 - 20,000 足够捕捉常用词和人名（如“罗辑”、“云天明”）。
    trainer = trainers.BpeTrainer(
        vocab_size=20000, 
        min_frequency=2,  # 至少出现2次才会被收录
        special_tokens=[
            # --- 基础控制符 ---
            "<|endoftext|>",  # 文档结束/EOS (End of Sentence)
            "<|padding|>",    # 填充符/PAD (Padding)
            
            # --- 对话/指令微调专用符 (ChatML风格) ---
            "<|im_start|>",   # 标记一句话的开始
            "<|im_end|>",     # 标记一句话的结束 (非常重要，防止模型自言自语停不下来)
            
            # --- 角色标识符 (显式占位) ---
            "<|system|>",     # 系统提示词 (System Prompt)
            "<|user|>",       # 用户输入
            "<|assistant|>",  # AI输出
            
            # --- 思考/思维链专用 (Optional, 类似 DeepSeek-R1) ---
            "<|thought|>",    # 开始思考
            "<|/thought|>"    # 结束思考
        ], # 特殊符号
        show_progress=True
    )

    # 5. 开始训练
    print(f"开始训练 Tokenizer，读取文件: {data_file} ...")
    tokenizer.train([data_file], trainer=trainer)

    # 6. 后处理 (Post-processing) - 可选
    # 在 BPE 之前通常不需要复杂的 post-processing，但在保存前最好确认一下
    
    # 7. 保存
    # 2. 【关键】如果文件夹不存在，必须先创建它！
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir,"threebody_tokenizer.json")
    tokenizer.save(save_path)
    print(f"训练完成！Tokenizer 已保存至: {save_path}")
    return tokenizer

# --- 执行训练 ---
# 请确保当前目录下有 three_body.txt 文件
if __name__ == "__main__":
    # 如果你有三个文件，可以传入列表：["part1.txt", "part2.txt", "part3.txt"]
    model_path = os.path.join(output_dir, "threebody_tokenizer.json")
    if not os.path.exists(model_path):
        tokenizer = train_threebody_tokenizer("./datasets/三体全集")
    else:
        print(f"🔍 发现已训练好的模型: {model_path}")
        print("📂 正在直接加载...")
        
        # --- 【核心代码】加载模型 ---
        tokenizer = Tokenizer.from_file(model_path)
        print("✅ 加载成功！")
    # --- 测试一下效果 ---
    test_text = "不要回答！不要回答！不要回答！这是叶文洁发出的警告。"
    encoded = tokenizer.encode(test_text)
    
    print("-" * 30)
    print(f"测试文本: {test_text}")
    print(f"分词结果 (Tokens): {encoded.tokens}")
    decoded_text = tokenizer.decode(encoded.ids)
    for i in encoded.ids:
        print(f"解码结果: {tokenizer.decode([i])}\n")
        
    print(f"对应的 IDs: {encoded.ids}")
    
    # 验证是否收录了专有名词
    name_test = "罗辑直接向三体世界发出了威慑。"
    encoded_name = tokenizer.encode(name_test)
    print(f"\n专有名词测试: {name_test}")
    print(f"分词结果: {encoded_name.tokens}")
    decoded_text = tokenizer.decode(encoded.ids)
    print(f"解码结果: {decoded_text}")
    # 观察 '罗辑' 是否被合并为一个 Token，还是分成了 '罗' 和 '辑'