# -*- coding: utf-8 -*-
"""
舆情助手 V51 修复版
1. 召回【官方后台跳转】按钮。
2. 保留【三维话术】与【心理洞察】核心功能。
3. 修复侧边栏布局。
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
FIXED_API_KEY = "" 

# ==========================================
# 1. 基础配置
# ==========================================

st.set_page_config(page_title="扇贝舆情话术舱 (V51)", layout="wide", page_icon="🐚")

@st.cache_resource
def load_ocr_model():
    return RapidOCR()

ocr = load_ocr_model()

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
# 2. 核心逻辑
# ==========================================

def clean_and_parse_json(text):
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
            temperature=0.8,
            response_format={ "type": "json_object" }
        )
        return clean_and_parse_json(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"API 调用失败: {str(e)}"}

# --- Prompt ---
PROMPT_V50 = """
你现在是【扇贝单词】的首席用户体验官（也是小红书文案大神）。
运营人员手动输入了一条用户的负面/咨询评论，请提供极致的回复策略。

【输入信息】
1. 用户评论：{user_text}
2. 内部事实(Context)：{context_info} (必须基于此事实进行解释或补偿，严禁胡编乱造)

【任务目标】
分析用户心理，并提供 3 种不同风格的回复，供运营根据当时语境选择。

【输出 JSON 结构】
{
    "insight": "一句话分析用户潜台词（例如：他其实不是想要退款，只是想要个解释/他现在极度愤怒，需要发泄窗口）",
    "options": {
        "style_soft": "方案A：软萌示弱型（适用于小Bug/日常吐槽。特点：叫宝宝，颜文字，替技术背锅，以此平息怒火）",
        "style_pro": "方案B：专业诚恳型（适用于功能失效/严肃建议。特点：不卑不亢，逻辑清晰，给出明确解决路径）",
        "style_humor": "方案C：幽默/自黑型（适用于非原则性槽点。特点：玩梗，把事故变故事，甚至能圈粉）"
    },
    "reply_dm": "私信引导话术（通用，目的是要ID或拉群，语气要急用户之所急）"
}
"""

# ==========================================
# 3. Streamlit UI 界面
# ==========================================

st.title("🐚 扇贝舆情话术舱 V51")
st.caption("针对已发现舆情 -> 生成高颗粒度回复方案")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 控制台")
    
    # API Key 逻辑
    if FIXED_API_KEY:
        api_key = FIXED_API_KEY
        st.success("✅ API Key 已就绪")
    else:
        if "DEEPSEEK_API_KEY" in st.secrets:
            api_key = st.secrets["DEEPSEEK_API_KEY"]
            st.success("✅ Secrets Loaded")
        else:
            api_key = st.text_input("DeepSeek Key", type="password")
    
    st.markdown("---")
    
    # ✅ 修复点：加回了官方后台跳转按钮
    st.link_button("🔗 打开官方反馈后台", "https://web.shanbay.com/words/app/feedback?shanbay_immersive_mode=true#/")
    
    st.markdown("---")
    st.markdown("### 📊 数据导出")
    if st.button("📥 导出今日处理记录 (CSV)"):
        if st.session_state.logs:
            df = pd.DataFrame(st.session_state.logs)
            st.download_button("点击下载 CSV", df.to_csv(index=False).encode('utf-8-sig'), "shanbay_replies.csv", "text/csv")
        else:
            st.warning("暂无记录")

# --- 主界面 ---
c1, c2 = st.columns([2, 3])

extracted_text = ""

with c1:
    st.markdown("##### 1. 捕获舆情")
    paste_result = paste_image_button(
        label="📋 粘贴截图 (Ctrl+V)",
        background_color="#3182ce",
        text_color="#ffffff",
        key="paste_v51"
    )
    
    if paste_result.image_data:
        st.image(paste_result.image_data, caption="截图预览", width=300)
        if st.button("🔍 提取文字"):
            with st.spinner("OCR 识别中..."):
                extracted_text = extract_text(paste_result.image_data)
    else:
        st.info("👈 点击左侧按钮粘贴截图，或直接在右侧输入")

with c2:
    st.markdown("##### 2. 话术生成配置")
    
    if extracted_text:
        st.session_state['v51_input'] = extracted_text
        
    user_text = st.text_area("用户评论内容", height=100, key="v51_input", placeholder="例如：你们新版背单词太卡了，会员白充了！")
    
    context_info = st.text_input(
        "🔧 内部事实/限制 (Context)", 
        placeholder="例如：技术已在修复预计10分钟好；无法退款但送7天会员...",
        help="AI 会基于此事实生成三种不同语气的文案。"
    )

    if st.button("✨ 生成三维话术方案", type="primary", disabled=not user_text):
        if not api_key:
            st.error("请配置 API Key")
        else:
            prompt = PROMPT_V50.replace("{user_text}", user_text).replace("{context_info}", context_info if context_info else "常规安抚")
            
            with st.spinner("正在揣摩用户心理并撰写文案..."):
                res = call_deepseek_api(prompt, user_text, api_key)
            
            if "error" in res:
                st.error(res["error"])
            else:
                st.divider()
                st.info(f"🧠 **心理洞察**：{res.get('insight')}")
                
                tab1, tab2, tab3 = st.tabs(["🥺 软萌示弱", "👔 专业诚恳", "🤡 幽默自黑"])
                options = res.get('options', {})
                
                with tab1:
                    st.code(options.get('style_soft'), language=None)
                with tab2:
                    st.code(options.get('style_pro'), language=None)
                with tab3:
                    st.code(options.get('style_humor'), language=None)
                
                st.markdown("---")
                st.markdown("**🤫 私信引导话术**")
                st.code(res.get('reply_dm'), language=None)
                
                st.session_state.logs.append({
                    "时间": time.strftime("%H:%M"),
                    "用户内容": user_text[:20],
                    "心理洞察": res.get('insight'),
                    "采纳方案": "待定" 
                })
