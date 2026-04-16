import streamlit as st
import sqlite3
import glob
import os
from google import genai
from google.genai import types
from pinecone import Pinecone
import time

# ==========================================
# ⚙️ 1. 초기 세팅 및 API 연결
# ==========================================
st.set_page_config(page_title="L-Bot", page_icon="⚖️", layout="wide")

# ==========================================
# 🎨 폰트 및 UI 스타일 설정 (아이콘 깨짐 방지 적용)
# ==========================================
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

/* 1. 앱 전체 기본 폰트를 프리텐다드로 적용 */
.stApp, p, span, div, h1, h2, h3, h4, h5, h6, li, a {
    font-family: 'Pretendard', sans-serif;
}

/* 2. 아이콘 폰트는 깨지지 않도록 원래 폰트를 강제 유지 (예외 처리) */
.material-icons, .material-symbols-rounded, .material-symbols-outlined {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# 보안 주의: 실제 서비스 시에는 st.secrets 사용 권장
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

@st.cache_resource
def init_clients():
    client = genai.Client(api_key=GEMINI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("legal-ai-db")
    return client, index

client, index = init_clients()
# ==========================================
# 🔒 보안: 비밀번호 확인 로직
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.warning("비밀번호를 입력하세요.")
        pwd = st.text_input("비밀번호", type="password")
        
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            st.rerun() # 정답이면 화면 새로고침하여 본 앱 실행
        elif pwd:
            st.error("비밀번호가 틀렸습니다.")
        return False
    return True

if not check_password():
    st.stop() # 비밀번호를 통과하지 못하면 여기서 코드를 멈춤!

# ==========================================
# 이후부터 기존 도구(Tools) 및 에이전트 로직 시작...
# ==========================================
# 🛠️ 2. 도구(Tools) 실구현 (수정 없음)
# ==========================================

def get_case_law(case_no: str) -> str:
    db_files = glob.glob("*_cases_*.db")
    if not db_files: return "DB 파일이 없습니다."
    for db in db_files:
        try:
            with sqlite3.connect(db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT summary, content FROM cases WHERE case_no LIKE ?", (f"%{case_no}%",))
                result = cursor.fetchone()
                if result:
                    # 💡 여기에 print를 추가하면, VScode나 CMD 터미널 창에 검색 결과가 뜹니다!
                    print(f"\n[DB 검색 성공!] {case_no} 찾음\n") 
                    return f"사건번호 {case_no} 찾음: {result[0]}\n본문: {result[1][:1000]}"
        except: continue
    return "판례를 찾지 못했습니다."

def search_cases_by_keyword(keyword: str) -> str:
    db_files = glob.glob("*_cases_*.db")
    res_list = []
    for db in db_files:
        try:
            with sqlite3.connect(db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT case_no, summary FROM cases WHERE content LIKE ? LIMIT 2", (f"%{keyword}%",))
                res_list.extend(cursor.fetchall())
        except: continue
    return str(res_list) if res_list else "관련 판례 없음"

def get_legal_theory(query: str) -> str:
    try:
        res_embed = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        results = index.query(vector=res_embed.embeddings[0].values, top_k=5, include_metadata=True)
        return "\n".join([f"[{m['metadata'].get('source')}] {m['metadata'].get('text')}" for m in results['matches']])
    except: return "교재 검색 실패"

# ==========================================
# 🤖 3. 에이전트 로직 (유연한 답변 및 템플릿 제거)
# ==========================================

st.title("⚖️ Law-Bot: 민형사 AI by 김성민")

sys_instruct = """
당신은 대한민국 최고의 법률 AI 에이전트입니다.
사용자의 질문 의도를 파악하여, 그에 맞는 형식으로 자유롭고 자연스럽게 답변하세요. 
일상적인 대화에는 부드럽게 응답하고, 판례나 법리를 설명할 때는 가독성 좋게 논리적으로 구성하십시오. 억지로 특정 템플릿에 맞출 필요는 없습니다.

[🚨 절대 규칙 1: 판례 지어내기(Hallucination) 엄격 금지]
1. 사용자가 특정 판례나 사건을 물어볼 경우, 반드시 'get_case_law' 또는 'search_cases_by_keyword' 도구를 사용하여 DB를 먼저 검색해야 합니다.
2. 도구를 통해 검색된 결과(Fact)가 아닌, 자신이 학습한 사전 지식이나 상상력을 동원하여 판례 번호, 사실관계, 판결 요지를 지어내는 행위를 절대 금지합니다.
3. 만약 도구 검색 결과 "판례를 찾지 못했습니다" 또는 "DB 파일이 없습니다" 등의 결과가 나온다면, 절대 말을 지어내지 말고 "현재 보유하고 있는 데이터베이스에서는 해당 판례를 찾을 수 없습니다."라고만 답변하십시오.

[🚨 절대 규칙 2: 교재 데이터 절단 대응]
교재 검색(get_legal_theory) 결과 문장이 불완전하게 잘려 있다면, 그 잘린 마지막 문장을 그대로 새로운 검색 쿼리(query)로 사용하여 'get_legal_theory'를 한 번 더 호출하세요. 이를 통해 끊긴 뒷부분의 문맥을 확보한 후 최종 답변을 작성해야 합니다. 절대 부족한 내용을 임의로 지어내지 마십시오.
"""

# 💡 세션 초기화: 코드가 꼬이지 않도록 여기서 딱 한 번만 Pro 모델로 세팅합니다!
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-3-flash-preview",  # 💎 할루시네이션을 잡기 위해 강력한 Pro 모델로 변경!
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct,
            tools=[get_case_law, search_cases_by_keyword, get_legal_theory],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
        )
    )

# 1. 메시지 초기화 및 화면 표시
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

# 2. 실행 트리거(Flag) 설정
need_generation = False

# 3. 재시도 버튼 (명확하고 안정적인 위치)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if st.button("🔄 에러 발생! 여기를 눌러 다시 답변 받기"):
        need_generation = True  

# 4. 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)
    
    if len(st.session_state.messages) > 10:
        st.session_state.messages = st.session_state.messages[-10:]
        
    need_generation = True  

# 5. AI 답변 생성 (Pro 모델, 3회 재시도 적용)
if need_generation:
    current_prompt = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        # 💡 상태창 메시지도 Pro 모델에 맞게 수정
        with st.status("💎 Pro 모델이 법률 데이터를 깊이 있게 분석 중입니다... (시도 1/3)", expanded=False) as status:
            
            final_answer = ""
            success = False
            max_retries = 3  
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        status.update(label=f"💎 서버 지연으로 재시도 중입니다... (시도 {attempt+1}/{max_retries})", state="running")
                        
                    response = st.session_state.chat_session.send_message(current_prompt)
                    
                    if response.text:
                        final_answer = response.text
                    else:
                        temp_parts = [part.text for part in response.candidates[0].content.parts if part.text]
                        if temp_parts:
                            final_answer = temp_parts[-1]
                    
                    if final_answer:
                        success = True
                        status.update(label="✅ 분석 완료!", state="complete")
                        break  
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1.5)
                        continue
                    else:
                        status.update(label=f"❌ 3회 시도 실패. (에러 원인: {str(e)})", state="error")
                        break

        # 6. 최종 답변 화면 출력 및 화면 새로고침
        if success and final_answer:
            st.caption("🤖 **💎 Pro** 모델이 답변을 생성했습니다.")
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.rerun()