# Portfolio — 어진희 (Eojinhee)

RAG · LLM 파인튜닝 · MLOps 전 영역을 직접 실험하고, 그 과정을 기록해 온 프로젝트 모음입니다.

- 📧 jinny040403@gmail.com
- 🔗 [github.com/Eojinhee](https://github.com/Eojinhee)

패스트캠퍼스 × Upstage AI Lab 24기 과정에서 RAG, LLM 파인튜닝, MLOps를 학습하며 6개 프로젝트(개인 1, 팀 5)를 수행했습니다. 모든 프로젝트에서 하이퍼파라미터·전처리 방식·모델 구조까지 모든 의사결정을 비교 실험으로 검증했습니다.

## 프로젝트 목록

| # | 프로젝트 | 기간 / 역할 | 핵심 성과 |
|---|---|---|---|
| 01 | [OCR/RS/AD 경진대회 — 이상탐지](./01-anomaly-detection) | 2026.06~진행중 · 개인 참가 | 홀드아웃 F1 1.0000 |
| 02 | [NLP 경진대회 — 대화 요약](./02-nlp-summarization) | 2026.05 · LLM 파인튜닝·Topic 생성 전략 | 최종 리더보드 1위 (ROUGE 52.56) |
| 03 | [냉장고 레시피 추천 RAG 챗봇](./03-recipe-rag-chatbot) | 2026.05 · Retrieval 파트 전담 | FAISS+MMR 기반 검색 파이프라인 구축 |
| 04 | [MLOps — TMDB 영화 평점 예측](./04-mlops-tmdb) | 2026.03 · 파이프라인/서빙 설계 참여 | RMSE 0.6332, 품질 게이트 자동화 |
| 05 | [ML 경진대회 — 서울 아파트 실거래가 예측](./05-ml-apartment-price) | 2026.02 · 파생변수 기획, 앙상블 | Private RMSE 36,876.34 |
| 06 | [IR 경진대회 — 한국어 과학지식 RAG 검색](./06-ir-rag-search) | 2026.05 · Hybrid 검색/RRF/쿼리 다각화 | MAP@3 0.9409 (79회 실험) |

## 기술 스택

**Language & Tools**: Python, Jupyter/JupyterLab, Pandas, Git/GitHub

**RAG / LLM**: LangChain, LangGraph, FAISS, BGE-M3, Cross-Encoder Reranking, Step-Back Prompting, Query Decomposition, RRF 앙상블, MMR

**Fine-tuning**: Qwen3-14B, LoRA(4bit Quantization, unsloth), KoBART, pko-t5-base, Solar API/Solar Pro

**MLOps**: Airflow(DockerOperator), FastAPI, Pydantic, MLflow, Weights & Biases

**ML**: LightGBM, CatBoostRegressor, PCA, 앙상블 모델링

## 프로젝트 요약

### 01. OCR/RS/AD 경진대회 — 이상탐지(Anomaly Detection)
화학 공정 센서 데이터 기반 비지도 이상탐지(Novelty Detection) 문제. PCA 기반 T²/SPE 통계량과 LightGBM을 결합해 홀드아웃 F1 1.0000 달성. Tennessee Eastman Process 외부 데이터를 0% 중복 검증 후 통합.

### 02. NLP 경진대회 — 대화 요약
KoBART baseline → Qwen3-14B LoRA SFT → Solar API 기반 Topic 증강 → MBR 앙상블 순으로 진행. 중간 리더보드 2위 → 최종 리더보드 1위(ROUGE 52.56)로 마무리.

### 03. 냉장고 레시피 추천 RAG 챗봇
KADX 만개의레시피(18.5만 건) 기반 RAG 챗봇. CSV 로드, 청킹, Upstage 임베딩+FAISS 벡터 DB, MMR 검색까지 Retrieval 파트를 전담.

### 04. MLOps — TMDB 영화 평점 예측 시스템
Docker 기반 7단계 파이프라인(수집→검증→전처리→학습→평가→품질게이트) 설계. 10만 건 중 82.6%가 노이즈임을 규명해 데이터 정제, FastAPI+Pydantic 기반 서빙까지 구현.

### 05. ML 경진대회 — 서울 아파트 실거래가 예측
Naver Geocoding API로 좌표 결측치 복원, 시계열 기반 파생변수 설계. CatBoostRegressor + Time-based Validation으로 Private RMSE 36,876.34 달성.

### 06. IR 경진대회 — 한국어 과학지식 RAG 검색
Elasticsearch 인덱싱, Hybrid(BM25+Dense) 검색, RRF 앙상블, Cross-Encoder Reranking 파이프라인 구축. 79회 실험 중 MAP@3 0.9409 달성.

## Contact

궁금한 점이나 협업 제안은 이메일로 연락 주세요: jinny040403@gmail.com


# OCR/RS/AD 경진대회 — 이상탐지(Anomaly Detection)

개인 참가 · 2026.06 ~ 진행중 · AD_이승윤 멘토님 팀

**담당**: 비지도 이상탐지(Unsupervised Novelty Detection) 모델링 — 개인 진행
**기술스택**: Tennessee Eastman Process(TEP) 벤치마크 데이터, 통계·ML·딥러닝·앙상블 기반 이상탐지

## 문제 정의

화학 공정 센서 데이터를 기반으로 이상 상태를 탐지하는 과제로, Tennessee Eastman Process(TEP) 벤치마크와 매칭되는 데이터입니다. 학습 데이터에 정상 상태만 포함되어 있어, 분류가 아닌 비지도 novelty detection 문제로 접근해야 한다는 점이 핵심입니다.

## 진행 전략

- 학습 데이터가 정상 샘플로만 구성된 비지도 문제임을 먼저 확인하고, 베이스라인 대비 가장 효과가 클 것으로 판단한 run 단위 집계(run-level aggregation)를 1순위 개선 포인트로 설정
- 통계 기반 → 지도학습 기반 → 딥러닝 기반 → 앙상블 순으로 3주 로드맵을 수립하고, 전 구간에 holdout 검증 절차를 적용

## 핵심 결과

- PCA 기반 T²/SPE 통계량 + 원본 피처 + 롤링 통계량을 결합한 피처셋으로 LightGBM 모델링
- **홀드아웃 F1 1.0000** 달성
- Tennessee Eastman Process 외부 데이터셋을 테스트 데이터와 0% 중복 검증 후 통합
- 총 63회 실험을 엑셀 로그(6개 시트)로 체계적으로 관리
- DPCA, PELT 기반 변화점 탐지 등 대안 기법도 비교했으나 성능 미달로 최종 방법론에서 제외

## 현재 상태

리더보드 오픈(2026.06.24), 제출 마감 2026.07.13. 진행 중인 프로젝트로 최종 결과는 추후 업데이트 예정입니다.


# NLP 경진대회 — 대화 요약(Dialogue Summarization)

Team Pyramid (5인: 김성빈·서민성·송용단·송현승·어진희) · 2026.05

**담당**: LLM 파인튜닝, Solar API 기반 Topic 생성 전략 설계, 앙상블 실험
**기술스택**: Qwen3-14B, LoRA(4bit Quantization, unsloth), KoBART, pko-t5-base, Solar API, ROUGE

## 문제 정의

대화문(dialogue)과 주제(topic)를 입력받아 한국어 요약문을 생성하는 과제로, ROUGE-1/2/L로 평가되었습니다. Train 12,457건 / Dev·Test 499건 규모였고, 화자 태그(#Person1# 등)와 핵심 사건 보존이 중요한 평가 포인트였습니다.

## 수행 과정

1. **KoBART baseline 구축** — 화자 태그 보존, 공백/줄바꿈 정리 등 데이터 정규화로 46.5 → 48.5 향상
2. **대체 모델 비교** — pko-t5-base(48.9), Solar API 자체 요약(44~45점대, KoBART보다 낮았으나 이후 Topic 생성 전략의 토대가 됨)
3. **Qwen3-14B 전환** — RTX 3090 한 대로 14B 모델 학습을 위해 4bit 양자화 LoRA 필수. unsloth의 num_beams 비호환, LoRA rank 16 초과 시 OOM, SFTTrainer eval_dataset 멀티프로세싱 오류 등을 직접 확인하며 안정적인 학습 구성 확정
4. **Topic 정보 결합** — Solar API로 Train/Dev/Test 전체에 Topic을 일관되게 생성해 학습에 반영 → 53.2~53.9점대로 도약
5. **MBR 앙상블** — 대회 종료 2시간 전, 제출 기회 5회 한도 내에서 LB 49점 이상 모델 19개를 모아 앙상블 구성

## 결과

**중간 리더보드 2위(ROUGE 53.91) → 최종 리더보드 1위(ROUGE 52.56)**로 마무리. 핵심 성공 요인은 모델 교체가 아니라 Topic 정보를 입력에 일관되게 포함시킨 것.

## 배운 점

좋은 요약은 모델 크기보다 입력 설계(무엇을 보여주는가)에 더 크게 좌우된다는 것을 수치로 확인했습니다. 제출 기회가 한정된 상황에서 신규 학습 대신 기존 산출물의 앙상블로 안정적인 개선을 끌어내는 의사결정을 경험했습니다.

# langchain_llmpj — 냉장고 레시피 추천 봇

냉장고 속 재료를 입력하면 어울리는 레시피를 추천해주는 RAG(Retrieval-Augmented Generation) 기반 챗봇입니다.

LLM 프로젝트 3조 · 심찬형 · 최유정 · 어진희 · 2026.06.16 발표

## 프로젝트 개요

**문제 상황**: 냉장고에 재료는 있는데 뭘 만들어야 할지 모르겠다 — 매번 레시피를 검색하기 번거롭고, 재료 조합에 맞는 레시피를 찾기 어려우며, 난이도/카테고리 필터링도 불편함

**해결 방법**: 재료를 입력하면 18만 건 이상의 레시피 데이터 기반 RAG로 맞춤 레시피를 추천하고, 난이도/카테고리/알레르기 필터와 대화형 후속 질문까지 지원

## 팀원 및 담당 역할

| 팀원 | 담당 |
|---|---|
| **어진희** | CSV 로드(18만 건 레시피), RecursiveCharacterTextSplitter 청킹, Upstage 임베딩 + FAISS 벡터 DB 구축, MMR 방식 유사 레시피 검색 |
| 최유정 | Pydantic 스키마 설계, RAG 체인 구성, JSON 구조화 출력 파서, 대화형 후속 질문(Memory), 난이도/카테고리/알레르기 필터 |
| 심찬형 | CLI 실행 흐름 구현, Streamlit 웹 UI, README·환경변수 관리, OpenAI/Upstage 혼재 API를 Upstage로 통일 |

## 데이터셋

- 출처: KADX 농식품 빅데이터 거래소(kadx.co.kr) — 만개의레시피
- 레시피 수: 약 18만 5천 개(2007~2023), 청크 수 184,172개
- 주요 컬럼: RCP_TTL(제목), CKG_MTRL_CN(재료), CKG_DODF_NM(난이도), CKG_TIME_NM(조리시간), CKG_STA_ACTO_NM(상황), CKG_MTH_ACTO_NM(조리방법)

## 아키텍처

1. 📄 데이터셋 로드 — 18만 건 레시피(KADX CSV)
2. ✂️ 청킹 — RecursiveCharacterTextSplitter
3. 🔢 임베딩 — Upstage solar-embedding-1-large
4. 🗄️ 벡터 DB 저장/검색 — FAISS + MMR 방식
5. 🤖 LLM 답변 생성 — Upstage Solar Pro

## 기술 스택

Python 3.10+ · LangChain / LangGraph · Upstage Solar Pro · Upstage Embeddings · FAISS Vector DB

## 주요 기능

- 🥕 재료 기반 레시피 추천 (3~6가지)
- ⭐ 난이도 선택 (초급/중급/고급 각 1개)
- 🍱 카테고리 필터 (일상/간식/술안주/손님접대/도시락, 최대 3개)
- 🚫 알레르기 재료 제외
- 💬 대화형 후속 질문 (Memory)

## 담당 파트 회고 (어진희)

> RAG 파이프라인의 Retrieval 파트를 전담하며, 청킹 전략·임베딩 모델 선택·MMR 검색 방식이 LLM의 최종 답변 품질에 직접적인 영향을 미친다는 것을 실감했습니다. 데이터 파이프라인이 LLM 서비스에서 얼마나 중요한 역할을 하는지 깨달았습니다.

## 개선 방향

- [ ] Streamlit UI 고도화
- [ ] 레시피 저장 기능 추가
- [ ] LangSmith 트레이싱 적용


# MLOps 팀 프로젝트 — TMDB 영화 평점 예측 시스템

6인 팀 프로젝트 (이지연·문장현·송병찬·어진희·박승수·김희수) · 2026.03

**기술스택**: Airflow(DockerOperator), FastAPI, Pydantic, MLflow, Weights & Biases, DNN+BatchNorm

## 파이프라인 설계

Step1~7(run_id 생성 → 수집 → 검증 → 전처리 → 학습 → 평가 → 품질게이트)을 독립 Docker 컨테이너로 분리, S3 URI로 단계 간 데이터 전달. DAG 완성이 지연되며 병목이 발생하자 MVP 직결형 파이프라인으로 전술을 전환해 배포 안정성과 속도를 동시에 확보. `model_bundle.tar.gz` 하나로 모델과 전처리기를 함께 패키징, SHA256 체크섬으로 무결성 검증.

## 데이터 진단 — "82.6%의 허상"

- 100건 → 1만 건 → 10만 건 순으로 단계적 전수 조사, 평점 분포의 왜도(Skewness 1.9527) 확인
- 10만 건 중 **82.6%(8.2만 건)가 투표수 50개 미만의 '유령 데이터'**임을 규명, vote_count<50 제거로 17,400건까지 정제
- Quantile Transform으로 분포 정상화

## 모델링 및 품질 게이트

- DNN + Batch Normalization + Dropout(0.3) → **RMSE 0.6332** 달성
- RMSE<1.2 미달 모델은 자동 차단되는 품질 게이트 자동화

## 서빙 및 협업

- FastAPI로 /health, /predict(단건), /predict-batch(배치) API 구현. Pydantic BaseModel(HealthResponse, PredictResponse 등)로 스키마 정의
- 예측 결과를 Parquet 저장 → 서버 메모리 로드로 응답 속도 확보
- MLflow(실험 기록)와 W&B(지표 비교) 역할 분리
- Data·Model·API·Pipeline 파트별 GitHub 이슈 카드로 협업 모듈화

## 향후 계획

model_bundle.tar.gz 단일 패키징 완료, 0.01초 이내 실시간 추론 엔진 전환 준비, 하이브리드 CI/CD 구조 지원

## 배운 점

"데이터의 82.6%가 노이즈"라는 결과처럼, 모델링 이전에 데이터 품질을 수치로 진단하는 과정이 성능 개선에 가장 직접적으로 기여한다는 것을 경험했습니다. 일정이 막힐 때 완벽한 구조를 고집하기보다 MVP로 먼저 동작시키고 이후 보완하는 전술 전환의 가치를 체감했습니다.

# ML 경진대회 — 서울 아파트 실거래가 예측

4조(사이좋게: 이지연·문장현·박승수·어진희·김희수) · 2026.02

**담당**: 파생변수 기획(초품아, 금리 등) 및 구현, 앙상블 모델링
**기술스택**: CatBoostRegressor, Naver Geocoding API, Time-based Validation

## 데이터 전처리

- 좌표 결측치를 단순 삭제하지 않고 Naver Geocoding API로 복원 — 도로명 주소 우선 검색 후 실패 시 지번 주소로 재검색, 병렬 처리로 속도 개선
- '해제사유발생일' 있는 비정상 거래 제거, 결측 80% 이상 컬럼 제거, 수치형 결측은 -999(이동평균 컬럼은 NaN 유지), 범주형은 'NA' 대체

## 피처 엔지니어링

- 아파트 단위 Lag(최근 3건) 가격, 구/시장 단위 이동평균(3·6개월) — 과거 데이터만 사용해 데이터 누수 방지
- 건물 노후도(building_age), 초등학교 인접 여부·금리 등 도메인 지식 기반 파생변수 직접 기획

## 모델링

- **CatBoostRegressor 선택 이유**: 범주형·수치형 혼합 데이터에 강함, Ordered Boosting으로 과적합 적음, 결측치/이상치에 강인, 튜닝 대비 재현성 높음
- Random K-Fold 대신 **Time-based Validation** 적용해 실제 평가 방식과 유사한 환경 구성
- learning rate·depth·iteration 튜닝, early stopping, log1p 타겟 변환

## 결과 및 회고

**Private RMSE 36,876.34** 기록. 시계열 기반 피처 설계가 성능 개선에 핵심적으로 기여. 다만 좌표 등 외부 데이터의 이상치(서울 주소인데 타 지역 좌표)가 성능을 악화시킨 것을 확인 — "외부 데이터는 추가보다 정합성 검증이 먼저"라는 원칙을 체득.

직접 파생변수를 기획하며 도메인 지식의 중요성을 느꼈고, 앙상블 기법으로 각 알고리즘의 장단점을 보완해 성능을 끌어올리는 모델링을 경험했습니다.

# IR 경진대회 — 한국어 과학지식 RAG 검색 시스템

IR_3조(최유정·심찬형·어진희) · 2026.05.19~06.03(2주) · MAP@3 평가

**담당**: Hybrid 검색(BM25+Dense) 및 RRF 융합 최적화, MultiQuery/HyDE/Step-Back 쿼리 다각화 전략, 다양한 Reranker 모델 비교, Function Calling 프롬프트 튜닝, 베이스라인 환경(Elasticsearch 인덱싱) 구축

**기술스택**: Elasticsearch, BGE-M3, RRF 앙상블, Cross-Encoder Reranking, Step-Back Prompting, Query Decomposition

## 대회 개요

- 문서 수: 4,272개(ko_mmlu + ko_ai2_arc)
- 쿼리 수: 220개(eval.jsonl)

## 파이프라인

① 의도 분류(GPT-4o-mini, 잡담/지식 구분) → ② 쿼리 확장(Step-Back + Decomposition) → ③ Chunk 검색(BGE-M3, Contextual Chunking) → ④ RRF 앙상블(MultiQuery 결과 통합) → ⑤ Reranking(BGE-Reranker, max_len=2048)

## 담당 역할

베이스라인 환경 셋업과 Elasticsearch 인덱싱 파이프라인 구축을 맡아 Colab 환경에서 발생하지 않던 설정 이슈를 GPU 서버 환경에서 직접 디버깅했습니다. Hybrid 검색(BM25+Dense)과 RRF 융합을 최적화했고, MultiQuery·HyDE·Step-Back 등 쿼리 다각화 전략을 실험했으며, 여러 Reranker 모델을 비교해 최적 조합을 찾았습니다.

## 결과

- 점수 추이: 0.3(베이스라인) → 0.61(BGE-M3 Dense+Reranker) → 0.83(의도분류 필터링) → 0.8818(MultiQuery+RRF) → 0.9394(Step-Back+Reranker2048) → **0.9409(앙상블, 최고점)**
- 최종 제출 점수: 0.9091
- 총 79회 실험 진행

**효과 있었던 기법**: 의도분류 필터링(+0.22), Step-Back+Decomposition(+0.10), Reranker max_length 2048, Contextual Chunking
**효과 없었던 기법**: BM25 Hybrid, HyDE, 도메인 필터링, LLM Reranker
