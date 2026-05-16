import os
import time
from llm_client import DEEPSEEK_V4_FLASH_LLMCLIENT, FileEmbedder

if __name__ == '__main__':
    try:
        llmClient = DEEPSEEK_V4_FLASH_LLMCLIENT()
        embedder = FileEmbedder("Qwen3-Embedding-0-6B")

        print("--- 向量化本地文件中 ---")
        start_time = time.monotonic_ns()
        results = embedder.embed_directory("国创赛")
        end_time = time.monotonic_ns()
        print(f"本次向量化共花费 {(end_time - start_time)/1e6:.2f} 毫秒")

        # ✅ 动态收集所有成功提取的文本
        file_contents = []
        for path, text, vec in results:
            if text:
                file_contents.append(f"【来源文件: {os.path.basename(path)}】\n{text}")

        if not file_contents:
            print("❌ 未提取到任何有效文本，终止运行。")
            exit()

        # ✅ 拼接为完整上下文
        combined_text = "\n\n---\n\n".join(file_contents)
        print(f"📊 待处理文本总长度: {len(combined_text)} 字符")

        # 保持你原有的 System Prompt
        system_prompt = (
            "你是一个严格遵循规则的项目信息提取Agent，专门从给定的项目标题列表中"
            "筛选出与人工智能技术相关的项目。并输出获奖届数（第x届“互联网+”、奖项等级、项目名称"
            "和项目负责人及项目所属大学及其AI领域关键词\n\n"
            "## 任务\n"
            "用户将提供一份项目标题列表，你需要逐一判断每个标题是否涉及"
            "以下一个或多个AI技术领域，"
            "并只输出符合条件的项目及对应领域标签。\n\n"
            "### 需要识别的领域及判定规则（精准匹配，忽略大小写，中英文同义）\n"
            "1. 计算机视觉：关键词包括"
            "计算机视觉、computer vision、CV、图像分类、目标检测、图像分割、"
            "语义分割、实例分割、目标跟踪、图像生成、图像增强、"
            "超分辨率、OCR、人脸识别、动作识别、"
            "姿态估计、深度估计、3D重建、GAN、扩散模型等（完整列表见下）。\n"
            "2. 机器视觉：机器视觉、machine vision、AOI、"
            "自动光学检测、工业相机等。\n"
            "3. 自然语言处理(NLP)：自然语言处理、"
            "NLP、文本分类、情感分析、NER、机器翻译、对话系统、"
            "BERT、GPT等。\n"
            "4. 多模态：多模态、multimodal、图文匹配、"
            "视觉问答、CLIP、Stable Diffusion等；"
            "若标题同时包含视觉和语言相关词也应视为多模态。\n"
            "5. 智能体(Agent)：智能体、agent、多智能体、"
            "LLM agent、工具学习、思维链、自我反思等。\n"
            "6. 大模型：大模型、foundation model、"
            "大语言模型、LLM、GPT、ChatGPT、文心一言、"
            "通义千问、参数规模、模型微调、RLHF等。\n\n"
            "### 排除规则\n"
            "- 标题含有“招聘”“培训”“课程”等非项目词且无强AI技术词，不予提取。\n"
            "- 仅含通用词“模型”“算法”而未指向上领域，不计入。\n\n"
            "### 输出格式\n"
            "严格按CSV格式输出，仅输出相关项目，"
            "每行包含：获奖年份、奖项等级、项目名称和项目负责人、"
            "项目所属大学及其AI领域关键词\n"
            "涉及AI领域若有多个，用分号隔开。无关项目直接跳过不输出。"
        )

        # ✅ 动态注入实际提取的数据
        user_prompt = (
            f"请处理以下大赛获奖名单数据，逐一判断并提取所有涉及AI领域的项目：\n\n{combined_text}"
        )

        exampleMessages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except Exception as e:
        print(f"发生错误：{e}，请重试！")
        import traceback
        traceback.print_exc()
