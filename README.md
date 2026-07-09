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
