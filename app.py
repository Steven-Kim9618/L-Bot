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
# 🎨 폰트 및 UI 스타일 설정 (Pretendard 적용)
# ==========================================
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Pretendard', sans-serif !important;
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

# 💡 수정됨: 강압적인 템플릿 요구를 지우고 상황에 맞게 대답하도록 지시
sys_instruct = """
당신은 대한민국 최고의 법률 AI 에이전트입니다.
사용자의 질문 의도를 파악하여, 그에 맞는 형식으로 자유롭고 자연스럽게 답변하세요. 
일상적인 대화에는 부드럽게 응답하고, 판례나 법리를 설명할 때는 가독성 좋게 논리적으로 구성하십시오. 억지로 특정 템플릿에 맞출 필요는 없습니다.

[🚨 중요 규칙: 교재 데이터 절단 대응]
교재 검색(get_legal_theory) 결과 문장이 불완전하게 잘려 있다면, 그 잘린 마지막 문장을 그대로 새로운 검색 쿼리(query)로 사용하여 'get_legal_theory'를 한 번 더 호출하세요. 이를 통해 끊긴 뒷부분의 문맥을 확보한 후 최종 답변을 작성해야 합니다. 절대 부족한 내용을 임의로 지어내지(Hallucination) 마십시오.
"""
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.5-flash", # 테스트 중엔 무조건 flash!
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct,
            
            # 💡 핵심 수정: 복잡한 설명서 다 지우고, 우리가 만든 진짜 파이썬 함수 3개를 그대로 넣습니다.
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

# 💡 2. 실행 트리거(Flag) 설정
# 이 변수가 True가 될 때만 AI가 답변을 생성하도록 통제하여 버튼 먹통 현상을 막습니다.
need_generation = False

# 💡 3. 재시도 버튼 (명확하고 안정적인 위치)
# 마지막 대화가 'user'의 질문으로 끝났다면 (즉, AI가 에러로 답변을 못 냈다면)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if st.button("🔄 에러 발생! 여기를 눌러 다시 답변 받기"):
        need_generation = True  # 버튼을 누르면 생성 트리거 ON

# 4. 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    # 질문을 저장하고 화면에 띄웁니다
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)
    
    # 토큰 절약 (최신 10개 유지)
    if len(st.session_state.messages) > 10:
        st.session_state.messages = st.session_state.messages[-10:]
        
    need_generation = True  # 새 질문이 들어와도 생성 트리거 ON

# 5. AI 답변 생성 (Flash 모델 전용, 3회 재시도 적용)
if need_generation:
    current_prompt = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        with st.status("⚡ Flash 모델이 법률 데이터를 분석 중입니다... (시도 1/3)", expanded=False) as status:
            
            final_answer = ""
            success = False
            max_retries = 3  # 💡 최대 3번 시도!
            
            if "chat_session" not in st.session_state:
                st.session_state.chat_session = client.chats.create(
                    model="gemini-2.5-flash",
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instruct,
                        tools=[get_case_law, search_cases_by_keyword, get_legal_theory],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
                    )
                )

            # 💡 3번 물고 늘어지는 반복문 시작
            for attempt in range(max_retries):
                try:
                    # 두 번째 시도부터는 상태창 글씨 업데이트
                    if attempt > 0:
                        status.update(label=f"⚡ 서버 지연으로 재시도 중입니다... (시도 {attempt+1}/{max_retries})", state="running")
                        
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
                        break  # 성공했으니 반복문 탈출!
                        
                except Exception as e:
                    # 에러가 났는데 아직 3번을 다 채우지 않았다면? -> 조금 쉬고 다시!
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1.5)
                        continue
                    else:
                        # 💡 3번 다 실패했다면? -> 어떤 에러인지 정확히 보여주기
                        status.update(label=f"❌ 3회 시도 실패. (에러 원인: {str(e)})", state="error")
                        break

        # 6. 최종 답변 화면 출력 및 화면 새로고침
        if success and final_answer:
            st.caption("🤖 **⚡ Flash** 모델이 답변을 생성했습니다.")
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.rerun()