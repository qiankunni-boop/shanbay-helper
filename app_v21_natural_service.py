# -*- coding: utf-8 -*-
"""
扇贝单词 · 智能舆情助手 (V47 Fixed Key)
核心升级：
1. 【免输 Key】：支持在代码头部配置固定 API Key，启动即用，拒绝重复劳动。
2. 【轻量稳定】：沿用 V46 的 RapidOCR + 无 iframe 稳定架构。
3. 【全能人设】：保留“宝宝体”、“替客服辩解”等所有高情商逻辑。
"""

import streamlit as st
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import numpy as np
from openai import OpenAI
import json
from streamlit_paste_button import paste_image_button

# ==========================================
# 0. 全局配置区 (在这里填入 Key)
# ==========================================

# 👇👇👇 请将您的 DeepSeek API Key 粘贴在下方引号内 👇👇👇
FIXED_API_KEY = "" 
# 例如：FIXED_API_KEY = "sk-99458a2eb9a3465886f3394d7ec6da69"

# ==========================================
# 1. 基础配置
# ==========================================

st.set_page_config(page_title="扇贝舆情助手 (V47 Fixed)", layout="wide")

@st.cache_resource
def load_ocr_model():
    return RapidOCR()

ocr = load_ocr_model()

def extract_text(image):
    try:
        img_array = np.array(image)
        result, _ = ocr(img_array)
        if not result:
            return ""
        texts = [line[1] for line in result]
        return " ".join(texts)
    except Exception as e:
        return f"识别出错: {str(e)}"

# ==========================================
# 2. DeepSeek AI 逻辑
# ==========================================

def call_deepseek_api(system_prompt, user_text, api_key):
    if not api_key:
        return {"error": "未检测到 API Key，请在代码头部配置或在侧边栏输入"}
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"用户评论内容：{user_text}"},
            ],
            stream=False,
            temperature=0.7, 
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"API 调用失败: {str(e)}"}

# --- 模式一：快速回复 ---
def analyze_fast_mode(text, api_key):
    prompt = """
    你现在是【扇贝单词】的贴心助教（人设：温柔、耐心的好朋友）。

    【🗣️ 核心话术风格】
    1. **强制开头**：必须以 **“好的，宝宝”** 或 **“宝宝消消气”** 开头。
    2. **特定句式（替客服说话）**：
       - 遇到进度慢/功能Bug：
       - “替我们客服辩解一下，这个功能确实在修了/记下来了，只是因为排期/优先级的问题，暂不清楚什么时候上线，所以可能还需要再等等。”
    3. **结尾要求**：必须包含歉意。

    【🚫 禁忌】
    - 严禁编造“底层架构”等虚假大词。
    - 不要像个机器人一样冷冰冰。

    【输出格式 (JSON)】
    {
        "scene": "...",
        "bug_type": "...",
        "reply_standard": "标准版回复 (60字内，按上述风格)",
        "reply_empathy": "共情版回复 (60字内，更软萌一点)"
    }
    """
    return call_deepseek_api(prompt, text, api_key)

# --- 模式二：深度分析 ---
def analyze_deep_mode(text, api_key):
    prompt = """
    你现在是【扇贝单词】的运营导师。请基于**“软性护短 + 诚恳示弱”**的人设提供思路。

    【任务 1：话术结构拆解】
    - Step 1: 情绪承接 (必须叫宝宝，先认错)
    - Step 2: 解释原因 (用“替客服辩解一下/排期问题”的逻辑)
    - Step 3: 收尾 (诚恳道歉)

    【任务 2：文案示范】
    写出符合以下风格的回复：
    “好的，宝宝...替客服辩解一下...非常抱歉...”

    【输出格式 (JSON)】
    {
        "user_emotion": "...",
        "structure_guide": [
            {"step": "1. 唤称与承接", "tips": "叫宝宝，接纳情绪..."},
            {"step": "2. 软性解释", "tips": "用排期/资源理由替团队辩解..."},
            {"step": "3. 诚恳收尾", "tips": "再次道歉..."}
        ],
        "reply_polished": "最终建议的回复文案"
    }
    """
    return call_deepseek_api(prompt, text, api_key)

# ==========================================
# 3. Streamlit UI 界面
# ==========================================

st.title("💖 扇贝舆情助手 (V47 Fixed)")
st.caption("状态：免输 Key 版 | 内核：RapidOCR Lite")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 控制台")
    
    # --- 核心修改：API Key 自动检测逻辑 ---
    if FIXED_API_KEY:
        api_key = FIXED_API_KEY
        st.success("✅ API Key 已从代码加载")
        st.caption(f"尾号: ...{FIXED_API_KEY[-4:]}")
    else:
        api_key = st.text_input("DeepSeek API Key", type="password")
        st.caption("💡 提示：在代码第25行填入 Key 可免输")
    
    st.markdown("---")
    st.markdown("### 🎛️ 模式切换")
    mode = st.radio(
        "选择功能模式",
        ["🚀 快速回复模式", "🧠 深度分析/润色"],
        captions=["日常 Bug 处理", "复杂吐槽/思维卡壳"]
    )
    
    st.markdown("---")
    st.link_button("🔗 打开官方反馈后台", "https://web.shanbay.com/words/app/feedback?shanbay_immersive_mode=true#/")

# ==========================================
# 模式 A：快速回复
# ==========================================
if mode == "🚀 快速回复模式":
    st.subheader("🚀 快速回复生成")
    
    c1, c2 = st.columns([1, 1])
    content = ""

    with c1:
        tab_paste, tab_upload = st.tabs(["📋 粘贴截图", "📂 上传图片"])
        with tab_paste:
            paste_result = paste_image_button(
                label="点此粘贴截图 (Ctrl+V)",
                background_color="#ff7875",
                hover_background_color="#ff4d4f",
                text_color="#ffffff",
                key="paste_fast"
            )
            if paste_result.image_data is not None:
                st.image(paste_result.image_data, width=280)
                if st.button("开始分析", key="btn_ocr_fast"):
                    with st.spinner("OCR 读取中..."):
                        content = extract_text(paste_result.image_data)
        with tab_upload:
            img_file = st.file_uploader("上传文件", type=["png", "jpg"], key="up_fast")
            if img_file:
                img = Image.open(img_file)
                st.image(img, width=280)
                if st.button("开始分析", key="btn_ocr_up_fast"):
                    content = extract_text(img)

    with c2:
        text_input = st.text_area("或直接粘贴文字", height=150, key="text_fast")
        if st.button("生成回复", key="btn_text_fast"):
            content = text_input

        if content:
            if not api_key:
                st.error("请先配置 API Key")
            else:
                st.divider()
                with st.spinner("DeepSeek 正在注入灵魂..."):
                    result = analyze_fast_mode(content, api_key)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    scene = result.get('scene', '未知')
                    bug_type = result.get('bug_type', '无')
                    st.markdown(f"**🎯 场景:** `{scene}` | **🔍 问题:** `{bug_type}`")
                    
                    st.info(f"**🔹 标准版:**\n{result.get('reply_standard')}")
                    st.success(f"**🔸 共情版:**\n{result.get('reply_empathy')}")

# ==========================================
# 模式 B：深度分析
# ==========================================
elif mode == "🧠 深度分析/润色":
    st.subheader("🧠 话术结构教练")

    user_input = st.text_area("在此粘贴让你头疼/卡壳的用户吐槽...", height=150)
    
    if st.button("✨ 帮我理清思路", key="btn_deep"):
        if not user_input:
            st.warning("请先输入内容")
        elif not api_key:
            st.error("请先配置 API Key")
        else:
            with st.spinner("正在拆解话术逻辑..."):
                result = analyze_deep_mode(user_input, api_key)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown(f"### 🌡️ 情绪诊断: `{result.get('user_emotion', '未知')}`")
                
                steps = result.get('structure_guide', [])
                cols = st.columns(len(steps))
                for i, step_data in enumerate(steps):
                    with cols[i]:
                        st.markdown(f"**{step_data['step']}**")
                        st.info(step_data['tips'])
                
                st.markdown("---")
                
                st.markdown("### ✍️ 建议回复示范")
                st.markdown(f"""
                <div style="background-color:#fff1f0; padding:20px; border-radius:10px; border-left: 5px solid #ff4d4f; color: #595959; font-size:16px;">
                    {result.get('reply_polished')}
                </div>
                """, unsafe_allow_html=True)
                st.text("")
                st.code(result.get('reply_polished'), language=None)
