# -*- coding: utf-8 -*-
"""
舆情助手 V53 终极全功能版
集成所有核心功能：
1. 双模式：【快速SOP话术】(三维方案) + 【深度逻辑拆解】(思维教练)。
2. 工具箱：OCR截图识别 + 事实注入(防幻觉) + 官方后台跳转 + CSV导出。
3. 稳定性：修复API Key语法 + 增强JSON解析。
"""

import streamlit as st
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import numpy as np
from openai import OpenAI
import json
import re
import pandas as pd
import time
from streamlit_paste_button import paste_image_button

# ==========================================
# 0. 全局配置区
# ==========================================

# 👇👇👇 请将您的 DeepSeek API Key 粘贴在下方 👇👇👇
FIXED_API_KEY = "sk-99458a2eb9a3465886f3394d7ec6da69" 
# (已修复引号闭合问题)

# ==========================================
# 1. 基础配置与缓存
# ==========================================

st.set_page_config(page_title="舆情话术库", layout="wide", page_icon="🐚")

@st.cache_resource
def load_ocr_model():
    return RapidOCR()

ocr = load_ocr_model()

# 初始化日志缓存
if 'logs' not in st.session_state:
    st.session_state.logs = []

@st.cache_resource
def get_openai_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def extract_text(image):
    try:
        img_array = np.array(image)
        result, _ = ocr(img_array)
        if not result: return ""
        texts = [line[1] for line in result]
        return " ".join(texts)
    except Exception as e:
        return f"识别出错: {str(e)}"

# ==========================================
# 2. 核心逻辑 (AI 交互)
# ==========================================

def clean_and_parse_json(text):
    """清洗 AI 返回的 Markdown 格式，提取 JSON"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try: return json.loads(match.group())
                except: pass
    return {"error": "JSON 解析失败", "raw_content": text}

def call_deepseek_api(system_prompt, user_text, api_key):
    if not api_key: return {"error": "❌ 未配置 API Key"}
    client = get_openai_client(api_key)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            stream=False,
            temperature=0.8, # 保持适度创造力
            response_format={ "type": "json_object" }
        )
        return clean_and_parse_json(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"API 调用失败: {str(e)}"}

# --- Prompt A: 快速话术 (SOP) ---
PROMPT_SOP = """
你现在是【扇贝单词】的小红书文案大神。
用户评论：{user_text}
内部事实(Context)：{context_info} (请基于此事实进行回复，若无则按常规处理)

请输出 JSON：
{
    "insight": "一句话分析用户潜台词（如：求安抚/求补偿/纯发泄）",
    "options": {
        "style_soft": "方案A(软萌示弱型)：叫宝宝+颜文字+替技术背锅",
        "style_pro": "方案B(专业诚恳型)：不卑不亢+逻辑清晰+解决方案",
        "style_humor": "方案C(幽默自黑型)：适度玩梗+拉近距离+化解尴尬"
    },
    "reply_dm": "私信引导话术（目的是要ID或拉入私域群）"
}
"""

# --- Prompt B: 深度拆解 (Logic Breakdown) ---
PROMPT_DEEP = """
你现在是【扇贝单词】的危机公关导师。
用户遇到了一个复杂/棘手的问题：{user_text}
请帮我拆解回复逻辑，一步步教我怎么回。

请输出 JSON：
{
    "emotion_diagnosis": "用户当前情绪状态诊断",
    "strategy_steps": [
        {"step": "Step 1: 情绪承接", "action": "具体怎么做"},
        {"step": "Step 2: 核心归因", "action": "怎么解释才得体"},
        {"step": "Step 3: 解决方案", "action": "给什么补偿或路径"}
    ],
    "final_reply": "综合上述逻辑的完整回复建议"
}
"""

# ==========================================
# 3. Streamlit UI 界面
# ==========================================

st.title("🐚小助手舆情辅助工具")

# --- 侧边栏：控制台 ---
with st.sidebar:
    st.header("⚙️ 控制台")
    
    # 1. API Key 检测
    if FIXED_API_KEY:
        api_key = FIXED_API_KEY
        st.success("✅ API Key 已内置")
    else:
        if "DEEPSEEK_API_KEY" in st.secrets:
            api_key = st.secrets["DEEPSEEK_API_KEY"]
            st.success("✅ Secrets 已加载")
        else:
            api_key = st.text_input("DeepSeek Key", type="password")
    
    st.markdown("---")
    
    # 2. 模式切换 (核心功能)
    mode = st.radio(
        "选择功能模式",
        ["🚀 快速话术生成 (SOP)", "🧠 深度逻辑拆解 (思维模式)"],
        captions=["日常高频：生成3种风格回复", "复杂危机：拆解步骤与逻辑"]
    )
    
    st.markdown("---")
    
    # 3. 官方后台跳转 (已保留)
    st.link_button("🔗 打开官方反馈后台", "https://web.shanbay.com/words/app/feedback?shanbay_immersive_mode=true#/")
    
    st.markdown("---")
    
    # 4. 数据导出 (已保留)
    st.markdown("### 📊 复盘数据")
    if st.button("📥 导出今日记录 (CSV)"):
        if st.session_state.logs:
            df = pd.DataFrame(st.session_state.logs)
            st.download_button("点击下载 CSV", df.to_csv(index=False).encode('utf-8-sig'), "shanbay_replies.csv", "text/csv")
        else:
            st.warning("暂无记录")

# ==========================================
# 模式 A：快速话术 (SOP)
# ==========================================
if mode == "🚀 快速话术生成 (SOP)":
    st.subheader("🚀 舆情话术库")
    
    c1, c2 = st.columns([2, 3])
    extracted_text = ""

    # 左侧：截图粘贴区
    with c1:
        st.info("步骤 1：获取内容")
        paste_result = paste_image_button(
            label="📋 粘贴截图 (Ctrl+V)",
            background_color="#3182ce",
            text_color="#ffffff",
            key="paste_sop"
        )
        if paste_result.image_data:
            st.image(paste_result.image_data, width=300)
            if st.button("🔍 提取文字", key="btn_ocr_sop"):
                with st.spinner("OCR 识别中..."):
                    extracted_text = extract_text(paste_result.image_data)
        else:
            st.caption("👈 点击蓝色按钮粘贴图片，或直接在右侧输入")

    # 右侧：生成配置区
    with c2:
        st.info("步骤 2：生成方案")
        
        # 自动回填 OCR 结果
        if extracted_text: st.session_state['input_sop'] = extracted_text
        
        user_text = st.text_area("用户评论", height=100, key="input_sop", placeholder="可以直接粘贴文字...")
        
        # 事实注入框 (保留)
        context_info = st.text_input("🔧 事实注入 (防止瞎编)", placeholder="例如：技术正在修复；无法退款但可送天数...")

        if st.button("✨ 生成三维话术", type="primary", disabled=not user_text):
            if not api_key: st.error("缺 API Key")
            else:
                prompt = PROMPT_SOP.replace("{user_text}", user_text).replace("{context_info}", context_info if context_info else "常规安抚")
                
                with st.spinner("正在生成三维话术方案..."):
                    res = call_deepseek_api(prompt, user_text, api_key)
                
                if "error" not in res:
                    st.divider()
                    st.success(f"🧠 **心理洞察**：{res.get('insight')}")
                    
                    t1, t2, t3 = st.tabs(["🥺 软萌示弱", "👔 专业诚恳", "🤡 幽默自黑"])
                    options = res.get('options', {})
                    
                    with t1: st.code(options.get('style_soft'), language=None)
                    with t2: st.code(options.get('style_pro'), language=None)
                    with t3: st.code(options.get('style_humor'), language=None)
                    
                    st.markdown("**🤫 私信引导话术：**")
                    st.code(res.get('reply_dm'), language=None)
                    
                    # 写入日志
                    st.session_state.logs.append({
                        "Time": time.strftime("%H:%M"), 
                        "Mode": "SOP", 
                        "Insight": res.get('insight'), 
                        "Content": user_text[:15]
                    })

# ==========================================
# 模式 B：深度逻辑拆解 (思维模式)
# ==========================================
elif mode == "🧠 深度逻辑拆解 (思维模式)":
    st.subheader("🧠 复杂舆情手术台")
    st.caption("适用场景：小作文、逻辑混乱、涉及多方责任，需要理清思路再回复。")
    
    deep_input = st.text_area("在此粘贴复杂的长难吐槽...", height=150, placeholder="用户写了一大段...")
    
    if st.button("🔪 开始逻辑拆解", type="primary"):
        if not api_key: st.error("缺 API Key")
        else:
            with st.spinner("正在抽丝剥茧..."):
                prompt = PROMPT_DEEP.replace("{user_text}", deep_input)
                res = call_deepseek_api(prompt, deep_input, api_key)
            
            if "error" not in res:
                st.divider()
                st.markdown(f"### 🌡️ 情绪诊断：`{res.get('emotion_diagnosis')}`")
                
                # 可视化步骤
                steps = res.get('strategy_steps', [])
                cols = st.columns(len(steps)) if steps else [st]
                for i, step in enumerate(steps):
                    with cols[i]:
                        st.markdown(f"**{step['step']}**")
                        st.info(step['action'])
                
                st.markdown("---")
                st.markdown("### ✍️ 建议回复")
                st.code(res.get('final_reply'), language=None)
                
                # 写入日志
                st.session_state.logs.append({
                    "Time": time.strftime("%H:%M"), 
                    "Mode": "Deep", 
                    "Insight": res.get('emotion_diagnosis'), 
                    "Content": deep_input[:15]
                })
