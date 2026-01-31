import os
import requests
import pyaudio
import threading
import queue
import re  # 增加正则用于清洗文本
from openai import OpenAI
import dotenv
import chromadb
import uuid
import time
import jieba  # 用于中文分词和实体识别
import jieba.posseg as pseg  # 词性标注

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
# --- 配置区 ---
env_vars = dotenv.dotenv_values(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
api_key = env_vars.get("ARK_API_KEY")
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
)
SOVITS_URL = "http://127.0.0.1:9880"
REF_AUDIO = r"E:\Local\GPT-SoVITS-v2pro-20250604-nvidia50\大家好，我是虚拟歌手洛天依.wav"
PROMPT_TEXT = "大家好，我是虚拟歌手洛天依，欢迎来到我的十周年生日会直播。"

# 初始化 ChromaDB 记忆系统
data_dir = r"E:\Local\ChromaDB\data"
os.makedirs(data_dir, exist_ok=True)
try:
    chroma_client = chromadb.PersistentClient(path=data_dir)
    memory_collection = chroma_client.get_or_create_collection(name="seeka_memory")
    memory_enabled = True
except Exception as e:
    print(f"[记忆系统初始化失败] {e}")
    print("记忆功能将被禁用。请检查网络连接或手动下载嵌入模型。")
    memory_collection = None
    memory_enabled = False

# 动态实体列表，用于逻辑覆盖
entity_keywords = set(["住址", "地址", "工作", "职业", "年龄", "姓名", "名字", "电话", "邮箱", "学校", "公司", "职位"])

# 更加严谨的切割符
PUNCTUATION = ['。', '！', '？', '!', '?', '\n', '；', ';', '：', ':', '，', ',']

# 对话历史缓存（最近5轮对话）
conversation_history = []

# 自动实体提取函数
def extract_entities(text):
    """从文本中自动提取可能的实体"""
    entities = set()
    
    # 使用jieba进行分词和词性标注
    words = pseg.cut(text)
    
    for word, flag in words:
        # 提取名词、人名、地名、机构名等
        if flag in ['nr', 'ns', 'nt', 'nz', 'n']:  # 人名、地名、机构名、其他名词
            if len(word) >= 1:  # 只考虑长度>=1的词
                entities.add(word)
        
        # 特殊规则：包含特定后缀的词
        if any(word.endswith(suffix) for suffix in ['公司', '大学', '医院', '学校', '银行', '政府', '中心', '局', '部']):
            entities.add(word)
        
        # 电话号码、邮箱等
        if re.match(r'\d{11}', word):  # 11位手机号
            entities.add(word)
        if '@' in word and '.' in word:  # 邮箱
            entities.add(word)
    
    return entities

# 扩展实体列表（限制数量，避免过度扩展）
def extend_entity_list(text):
    """从文本中提取新实体并添加到全局列表"""
    global entity_keywords
    new_entities = extract_entities(text)
    # 只添加与当前对话相关的实体，避免无限增长
    relevant_entities = set()
    for entity in new_entities:
        if entity in text and len(entity) > 1:  # 确保实体在当前文本中且长度合适
            relevant_entities.add(entity)
    
    entity_keywords.update(relevant_entities)
    if relevant_entities:
        print(f"[实体扩展] 新增实体: {list(relevant_entities)}")
    
    # 限制实体列表大小，避免内存溢出
    if len(entity_keywords) > 100:
        # 移除最旧的实体（简单策略：移除前20个）
        entity_keywords = set(list(entity_keywords)[20:])

# 清理记忆函数：删除存入超过7天且访问次数<2的记忆
def cleanup_memory():
    if not memory_enabled or not memory_collection:
        return
    try:
        results = memory_collection.get(include=["metadatas"])
        current_time = time.time()
        to_delete = []
        for i, metadata in enumerate(results["metadatas"]):
            if metadata:
                timestamp = metadata.get("timestamp", 0)
                access_count = metadata.get("access_count", 0)
                if current_time - timestamp > 7 * 24 * 3600 and access_count < 2:
                    to_delete.append(results["ids"][i])
        if to_delete:
            memory_collection.delete(ids=to_delete)
            print(f"[记忆清理] 删除了 {len(to_delete)} 条不重要的记忆")
    except Exception as e:
        print(f"[记忆清理错误] {e}")

# 逻辑覆盖机制：Single Source of Truth
def apply_logical_override(conversation):
    if not memory_enabled or not memory_collection:
        return
    
    # 先扩展实体列表
    extend_entity_list(conversation)
    
    # 检查是否包含已知实体，且实体在当前对话中出现多次（表示重要性）
    important_entities = []
    for entity in entity_keywords:
        if conversation.count(entity) > 1:  # 实体在对话中出现多次
            important_entities.append(entity)
    
    if not important_entities:
        return  # 没有重要实体，不进行覆盖
    
    # 只对重要实体进行覆盖
    for entity in important_entities[:2]:  # 限制处理实体数量
        try:
            # 搜索包含该实体的旧记忆
            results = memory_collection.query(
                query_texts=[entity],
                n_results=5,  # 减少搜索结果
                include=["documents", "metadatas"]
            )
            if results.get('documents') and results['documents'][0]:
                to_delete = set()
                current_time = time.time()
                for i, doc in enumerate(results['documents'][0]):
                    if entity in doc and i < len(results['ids'][0]):
                        # 只删除超过1小时且访问次数少的记忆
                        metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else None
                        if metadata and current_time - metadata.get("timestamp", 0) > 3600 and metadata.get("access_count", 0) < 3:
                            to_delete.add(results['ids'][0][i])
                to_delete = list(to_delete)
                if to_delete:
                    memory_collection.delete(ids=to_delete)
                    print(f"[逻辑覆盖] 删除了 {len(to_delete)} 条关于'{entity}'的旧记忆")
        except Exception as e:
            print(f"[逻辑覆盖错误] {e}")

text_queue = queue.Queue()
audio_queue = queue.Queue()

def clean_text(text):
    """清除表情符号和多余特殊字符，防止模型卡死"""
    # 移除表情符号 (Emoji)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s' + "".join(PUNCTUATION) + r']', '', text)
    # 将多个连续空格换成一个，防止模型停顿过久
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tts_request_worker():
    while True:
        text = text_queue.get()
        if text is None: break
        
        cleaned_text = clean_text(text)
        if not cleaned_text or len(cleaned_text) < 1:
            text_queue.task_done()
            continue

        try:
            tts_data = {
                "text": cleaned_text,
                "text_lang": "zh",  # 必须开启自动识别
                "ref_audio_path": REF_AUDIO,
                "prompt_lang": "zh",
                "prompt_text": PROMPT_TEXT,
                "text_split_method": "cut5",
                "media_type": "raw", 
                "streaming_mode": 1,
                "parallel_infer": True # 开启并行推理，压榨5060性能
            }
            with requests.post(f"{SOVITS_URL}/tts", json=tts_data, stream=True, timeout=8) as resp:
                if resp.status_code == 200:
                    for chunk in resp.iter_content(chunk_size=2048):
                        if chunk:
                            audio_queue.put(chunk)
        except Exception as e:
            print(f"\n[TTS请求异常] {e}")
        text_queue.task_done()

def playback_worker():
    p = pyaudio.PyAudio()
    # 采样率确保匹配模型。如果声音不对，请尝试将 32000 改为 24000 或 44100
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=32000, output=True, frames_per_buffer=1024)
    while True:
        chunk = audio_queue.get()
        if chunk is None: break
        stream.write(chunk)
        audio_queue.task_done()

threading.Thread(target=tts_request_worker, daemon=True).start()
threading.Thread(target=playback_worker, daemon=True).start()

def main():
    # 启动时清理记忆
    cleanup_memory()
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        # 更新对话历史
        global conversation_history
        conversation_history.append(f"用户: {user_input}")
        if len(conversation_history) > 10:  # 保持最近10轮
            conversation_history = conversation_history[-10:]
        
        # 检索相关记忆（基于对话历史而非单个输入）
        if memory_enabled and memory_collection:
            try:
                # 使用最近的对话历史作为查询上下文
                query_context = " ".join(conversation_history[-3:])  # 使用最近3轮对话
                results = memory_collection.query(
                    query_texts=[query_context], 
                    n_results=2,  # 减少结果数量，避免混淆
                    include=["documents", "metadatas"]
                )
                relevant_memories = results.get('documents', [[]])[0] if results.get('documents') else []
                # 过滤掉过于旧的记忆（超过1小时）
                current_time = time.time()
                filtered_memories = []
                for i, doc in enumerate(relevant_memories):
                    if results.get('metadatas') and results['metadatas'][0] and i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        if metadata and current_time - metadata.get("timestamp", 0) < 3600:  # 1小时内
                            filtered_memories.append(doc)
                
                memory_context = "\n".join(filtered_memories) if filtered_memories else "无相关记忆。"
                
                # 更新访问次数
                if results.get('ids') and results['ids'][0]:
                    for i, doc_id in enumerate(results['ids'][0][:len(filtered_memories)]):  # 只更新保留的记忆
                        if results.get('metadatas') and results['metadatas'][0] and i < len(results['metadatas'][0]):
                            metadata = results['metadatas'][0][i]
                            if metadata:
                                current_count = metadata.get("access_count", 0)
                                memory_collection.update(
                                    ids=[doc_id], 
                                    metadatas=[{"access_count": current_count + 1}]
                                )
            except Exception as e:
                print(f"[记忆检索错误] {e}")
                memory_context = "无相关记忆。"
        else:
            memory_context = "记忆功能未启用。"
        
        # 预热：发送2块静音包，彻底唤醒声卡驱动
        audio_queue.put(b'\x00' * 4096)
        audio_queue.put(b'\x00' * 4096)

        print("Local: ", end="", flush=True)
        try:
            system_prompt = os.getenv("SYSTEM_PROMPT", "你是一个有帮助的AI助手。")
            messages = [
                {"role": "system", "content": f"{system_prompt}\n\n相关记忆：\n{memory_context}"},
                {"role": "user", "content": user_input},
            ]
            stream = client.chat.completions.create(
                model="deepseek-v3-250324", 
                messages=messages,
                stream=True,
            )

            current_sentence = ""
            first_chunk_triggered = False 
            full_response = ""

            for chunk in stream:
                if not chunk.choices: continue
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_response += content
                    current_sentence += content
                    
                    # 1. 针对首句的极致优化：只要出到5个字且包含空格或标点，立即切
                    if not first_chunk_triggered and len(current_sentence) >= 6:
                        if any(p in current_sentence for p in PUNCTUATION) or ' ' in content:
                            text_queue.put(current_sentence)
                            current_sentence = ""
                            first_chunk_triggered = True
                            continue

                    # 2. 正常切句
                    if any(p in content for p in PUNCTUATION):
                        if len(current_sentence.strip()) > 1:
                            text_queue.put(current_sentence)
                            current_sentence = ""
                            first_chunk_triggered = True

            if current_sentence.strip():
                text_queue.put(current_sentence)

            # 存储对话到记忆
            if memory_enabled and memory_collection:
                try:
                    conversation = f"用户: {user_input}\nAI: {full_response}"
                    # 应用逻辑覆盖机制
                    apply_logical_override(conversation)
                    memory_collection.add(
                        documents=[conversation], 
                        metadatas=[{"timestamp": time.time(), "access_count": 0}], 
                        ids=[str(uuid.uuid4())]
                    )
                    # 更新对话历史
                    conversation_history.append(f"AI: {full_response}")
                except Exception as e:
                    print(f"\n[记忆存储错误] {e}")

        except Exception as e:
            print(f"\n[LLM错误] {e}")

if __name__ == "__main__":
    main()