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
# 🤖 3. 에이전트 로직 (None 방지 보강)
# ==========================================

st.title("⚖️ Law-Bot: 민형사 AI by 김성민")

# 시스템 지시문 강화: 답변을 끝까지 완성하라고 명시
sys_instruct = """
당신은 대한민국 최고의 법률 AI 에이전트입니다.
질문을 받으면 도구를 사용해 정보를 검색한 뒤, 반드시 아래의 마크다운 템플릿 구조를 그대로 복사하여 빈칸을 채우는 방식으로만 답변을 출력하세요. 
다른 인사말이나 서론 없이 무조건 '### 1. 사건 개요 및 판결 요지'부터 출력을 시작해야 합니다.

[🚨 중요 규칙: 교재 데이터 절단 대응]
교재 검색(get_legal_theory) 결과 문장이 불완전하게 잘려 있다면, 그 잘린 마지막 문장을 그대로 새로운 검색 쿼리(query)로 사용하여 'get_legal_theory'를 한 번 더 호출하세요. 이를 통해 끊긴 뒷부분의 문맥을 확보한 후 최종 답변을 작성해야 합니다. 절대 부족한 내용을 임의로 지어내지(Hallucination) 마십시오.

### 1. 사건 개요 및 판결 요지
- **사건번호:** ('get_case_law' 도구 검색 결과)
- **사실관계:** (사건이 어떻게 발생했는지 구체적으로 기재)
- **법원의 판단:** (인정된 범죄와 최종 판결 요지)

### 2. 관련 핵심 법리 (교재 기준)
- ('get_legal_theory' 도구 검색 결과를 바탕으로, 이 사건에 적용된 핵심 이론 상세 설명)

### 3. 이론의 구체적 적용
- (1번의 사실관계에 2번의 이론이 어떻게 적용되어 저런 판결이 나왔는지 종합적으로 논술)
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

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("전문 지식을 분석 중입니다..."):
            
            # 1. 사용할 모델 후보 리스트 설정
            models_to_try = ["gemini-2.5-pro", "gemini-2.5-flash"]
            max_retries_per_model = 2 # 각 모델당 2번씩 시도 (총 4번)
            final_answer = ""
            active_model = ""

            # 2. 모델 리스트를 순회하며 시도
            for model_name in models_to_try:
                active_model = model_name
                success = False
                
                for attempt in range(max_retries_per_model):
                    try:
                        # 현재 시도 중인 모델로 세션의 모델 정보를 임시 변경하여 전송
                        # (chat_session 생성 시 모델이 고정되어 있으므로, 
                        # send_message 시점에 config를 통해 모델을 지정하거나 세션을 새로 파야 할 수 있지만, 
                        # 여기서는 단순화를 위해 세션 생성 시의 모델 변수를 활용하는 방식으로 설명합니다.)
                        
                        # 💡 팁: chat_session의 모델을 직접 바꿀 수 없으므로 
                        # 실제로는 세션을 재생성하거나 send_message의 개별 호출 방식을 씁니다.
                        # 가장 확실한 방법은 아래처럼 '모델' 이름만 바꿔서 재호출하는 것입니다.
                        
                        response = st.session_state.chat_session.send_message(
                            prompt,
                            config=types.GenerateContentConfig(model=active_model) # 모델 동적 지정
                        )
                        
                        # 응답 텍스트 추출
                        if response.text:
                            final_answer = response.text
                        else:
                            for candidate in response.candidates:
                                for part in candidate.content.parts:
                                    if part.text: final_answer += part.text
                        
                        if final_answer:
                            success = True
                            break # 시도 루프 탈출
                            
                    except Exception as e:
                        if "503" in str(e) or "429" in str(e):
                            time.sleep(1.5) # 서버 부하 시 잠시 대기
                            continue
                        else:
                            st.error(f"오류 발생: {e}")
                            break
                
                if success: break # 모델 순회 루프 탈출

            # 3. 화면 출력
            if final_answer:
                # 어떤 모델이 응답했는지 상단에 작게 표시
                model_label = "💎 Pro" if "pro" in active_model else "⚡ Flash"
                st.caption(f"🤖 **{model_label}** 모델이 답변을 생성했습니다.")
                
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            else:
                st.error("현재 모든 AI 서버가 응답하지 않습니다. 잠시 후 다시 시도해 주세요.")
