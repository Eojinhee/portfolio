# 🚀 Portfolio — 어진희 (Eojinhee)

> RAG · LLM 파인튜닝 · MLOps 전 영역을 직접 실험하고, 그 과정을 기록해 온 프로젝트 모음입니다.
> - 📧 **Email**: jinny040403@gmail.com
> - 🔗 **GitHub**: [github.com/Eojinhee](https://github.com/Eojinhee)
> - 🎓 **Education**: 패스트캠퍼스 × Upstage AI Lab 24기 수료 | 신구대학교 AI소프트웨어과 졸업

---

## 🛠️ Tech Stack & Core Competencies

- **Language & Tools**: Python, Jupyter / JupyterLab, Pandas, Git / GitHub
- **RAG / LLM**: LangChain, LangGraph, FAISS, Elasticsearch, BGE-M3, Cross-Encoder Reranking, Step-Back Prompting, Query Decomposition, RRF 앙상블, MMR
- **Fine-tuning & Models**: Qwen3-14B, LoRA (4bit Quantization, unsloth), KoBART, pko-t5-base, Solar API / Solar Pro
- **MLOps & Infra**: Docker, Airflow (DockerOperator), FastAPI, Pydantic, MLflow, Weights & Biases
- **Machine Learning**: LightGBM, CatBoostRegressor, PCA, 앙상블 모델링

---

## 📅 프로젝트 목록 (Total 6 Projects)

| 번호 | 프로젝트 명 | 기간 / 역할 | 핵심 성과 |
| :---: | :--- | :---: | :--- |
| **01** | **OCR/RS/AD 경진대회 — 이상탐지** | 2026.06 ~ 07 · 개인 | **홀드아웃 F1 1.0000** (LB F1 0.9520, 63회 실험) |
| **02** | **NLP 경진대회 — 대화 요약** | 2026.05 · LLM 파인튜닝/Topic 전략 | **최종 리더보드 1위 (ROUGE 52.56)** |
| **03** | **냉장고 레시피 추천 RAG 챗봇** | 2026.05 · Retrieval 전담 | 18.5만 건 FAISS + MMR 기반 검색 파이프라인 구축 |
| **04** | **MLOps — TMDB 영화 평점 예측** | 2026.03 · 파이프라인/서빙 설계 | **82.6% 유령 데이터 정제**, RMSE 0.6332, 품질 게이트 |
| **05** | **ML 경진대회 — 서울 아파트 실거래가 예측** | 2026.02 · 파생변수 기획/앙상블 | Naver Geocoding 좌표 복원, Private RMSE 36,876.34 |
| **06** | **IR 경진대회 — 한국어 과학지식 RAG 검색** | 2026.05 · Hybrid/RRF/쿼리 다각화 | **MAP@3 0.9409** (79회 실험) |

---

## 📂 프로젝트 상세 내용

### 01. OCR/RS/AD 경진대회 — 이상탐지 (Anomaly Detection)
> **개인 참가 · 2026.06 ~ 2026.07 · AD_이승윤 멘토님 팀**
> **담당**: 비지도 이상탐지 모델링 — 개인 진행
> **기술 스택**: TEP 벤치마크 데이터, 통계·ML·딥러닝·앙상블 기반 이상탐지

- **문제 정의**:
  - 화학 공정 센서 데이터를 기반으로 이상 상태를 탐지하는 과제로, 학습 데이터에 정상 상태만 포함되어 있어 단순 분류가 아닌 비지도 novelty detection 관점으로 접근해야 했습니다.
- **진행 전략**:
  - 베이스라인의 Row 단위 판단 구조가 가진 한계를 돌파하기 위해, 전체 가동 단위를 아우르는 Run 단위 집계를 핵심 개선 포인트로 잡고 3주간 로드맵을 수행했습니다.
- **모델 채택 이유 및 적용 효과**:
  - **PCA (주성분 분석)**
    - **채택 이유**: 고차원 센서 변수들 사이에 존재하는 다중공선성 문제를 해결하고, 정상 데이터가 이루는 주성분 공간에서 얼마나 벗어나는지를 수치화하기 위해 선택했습니다.
    - **적용 효과**: $T^2$ 및 SPE 통계량을 바탕으로 Train 정상 데이터의 99분위수를 기준점으로 삼아 리더보드 상위권으로 도약할 수 있는 발판을 마련했습니다.
  - **LightGBM**
    - **채택 이유**: 비지도 단계에서 뽑아낸 통계적 특징과 시계열 롤링 값을 결합해 고장 여부를 판정할 때, 고속 학습이 가능하고 연속형 데이터 처리에 유리하다고 판단했습니다.
    - **적용 효과**: 홀드아웃 F1 1.0000을 달성했고, 비지도 PCA와 완전히 다른 접근법임에도 결과적으로 동일한 215개의 고장 Run을 정확히 도출해 내며 교차 검증을 완수했습니다. (총 63회 실험)

---

### 02. NLP 경진대회 — 대화 요약 (Dialogue Summarization)
> **Team Pyramid (5인) · 2026.05**
> **담당**: LLM 파인튜닝, Solar API 기반 Topic 생성 전략 설계, MBR 앙상블 실험
> **기술 스택**: Qwen3-14B, LoRA (4bit Quantization, Unsloth), KoBART, pko-t5-base, Solar API, ROUGE

- **문제 정의**:
  - 대화문과 주제를 받아 한국어 요약문을 정확하게 생성하는 과제 (Train 12,457건)로, 화자 태그와 핵심 사건의 누락 없는 보존이 관건이었습니다.
- **수행 과정 및 모델 채택 이유 & 효과**:
  - **KoBART 및 pko-t5-base 도입 후 전환**
    - **이유**: 가볍고 빠른 학습을 위해 먼저 적용해 보았으나, 복잡한 대화 흐름을 매끄럽게 잇거나 문맥을 장기적으로 유지하는 측면에서 한계가 뚜렷해 고성능 오픈소스 LLM으로 선회했습니다.
  - **Qwen3-14B & LoRA (4bit Quantization, Unsloth)**
    - **채택 이유**: 단일 RTX 3090 환경이라는 하드웨어적 제약 속에서 14B 규모의 대형 모델을 안정적으로 학습시켜야 했습니다. 4bit 양자화와 unsloth 프레임워크를 결합해 OOM 문제를 원천 차단하면서 모델 본연의 언어 이해력을 최대한 끌어내고자 채택했습니다.
    - **적용 효과**: 학습 안정성을 확보하고 고성능 파인튜닝 파이프라인을 구축했습니다.
  - **Solar API Topic 증강 도입**
    - **채택 이유**: 단순히 모델만 키워서는 성능 향상에 한계가 있어, 대화 전체를 관통하는 주제(Topic)를 추출해 입력에 주입하는 전처리 전략이 필수적이라 판단했습니다.
    - **적용 효과**: 점수가 53점대로 급상승하며 최종 리더보드 1위(ROUGE 52.56)를 기록했습니다. (19개 모델 MBR 앙상블 적용)

---

### 03. langchain_llmpj — 냉장고 레시피 추천 봇
> **LLM 프로젝트 3조 · 2026.06.16 발표**
> **담당**: CSV 로드 (18만 건), RecursiveCharacterTextSplitter 청킹, Upstage 임베딩 + FAISS 벡터 DB 구축, MMR 유사 검색
> **기술 스택**: Python 3.10+ · LangChain / LangGraph · Upstage Solar Pro · Upstage Embeddings · FAISS Vector DB

- **프로젝트 개요**:
  - 농식품 빅데이터 거래소의 18.5만 건 레시피 데이터를 활용해 사용자의 냉장고 재료에 꼭 맞는 맞춤형 레시피를 제안하는 RAG 챗봇 구축.
- **기술 채택 이유 및 적용 효과**:
  - **FAISS & MMR (Maximal Marginal Relevance)**
    - **채택 이유**: 18만 건이 넘는 방대한 레시피를 조회할 때 단순 코사인 유사도 방식만 쓰면 대중적이고 비슷한 메뉴만 계속 추천되는 한계가 있었습니다. 검색 정확도를 지키면서도 결과 간의 다양성을 확보하기 위해 도입했습니다.
    - **적용 효과**: 사용자가 입력한 재료 조합에 맞춰 서로 다른 카테고리의 신선하고 다양한 레시피를 폭넓게 추천할 수 있게 되었습니다.
  - **Upstage solar-embedding-1-large**
    - **채택 이유**: 범용 임베딩 모델을 테스트해 본 결과, 한국어 조리법 특유의 비정형 표현이나 재료명 수식어 매칭에 다소 아쉬움이 있어 교체를 결정했습니다.
    - **적용 효과**: 한국어 문맥 이해도가 뛰어난 임베딩 모델로 전환한 덕분에 재료와 레시피 간의 의미적 유사도 매칭 품질이 크게 높아졌습니다.

---

### 04. MLOps 팀 프로젝트 — TMDB 영화 평점 예측 시스템
> **6인 팀 프로젝트 · 2026.03**
> **담당**: Airflow 파이프라인 구축, 데이터 진단, FastAPI 서빙 및 패키징
> **기술 스택**: Airflow (DockerOperator), FastAPI, Pydantic, MLflow, Weights & Biases, DNN + BatchNorm

- **모델 및 인프라 채택 이유 & 효과**:
  - **DNN + BatchNorm + Dropout (0.3)**
    - **채택 이유**: 수치형 데이터와 범주형 임베딩 피처가 혼재된 영화 메타데이터의 복잡한 비선형 관계를 유연하게 학습시키기 위해 다층 퍼셉트론 구조를 선택했습니다. 아울러 학습 과정의 불안정성을 잡기 위해 배치 정규화를 추가했습니다.
    - **적용 효과**: 철저하게 정제된 데이터셋과 맞물려 RMSE 0.6332의 우수한 예측 성능을 달성했으며, 자동화된 품질 검증 단계를 통과했습니다.
  - **Airflow + DockerOperator**
    - **채택 이유**: 데이터 수집부터 최종 모델 검증에 이르는 전체 파이프라인의 각 단계를 독립된 컨테이너로 격리하고, 단계별 의존성을 코드로 명확하게 관리하기 위해 채택했습니다.
    - **적용 효과**: 일정 지연 위기 속에서 MVP 중심의 빠른 전술 전환을 이뤄내며 배포 안정성과 파이프라인 실행 속도를 동시에 사수했습니다. (82.6% 유령 데이터 정제 병행)

---

### 05. ML 경진대회 — 서울 아파트 실거래가 예측
> **4조 · 2026.02**
> **담당**: 파생변수 기획, 외부 데이터 연동, 앙상블 모델링
> **기술 스택**: CatBoostRegressor, Naver Geocoding API, Time-based Validation

- **모델 채택 이유 및 적용 효과**:
  - **CatBoostRegressor**
    - **채택 이유**: 아파트 실거래가 데이터는 지역이나 동 같은 범주형 정보와 면적, 금리 같은 수치형 정보가 복잡하게 얽혀 있습니다. CatBoost는 범주형 변수를 효율적으로 처리하고 순차적 부스팅을 통해 과적합을 강하게 억제하므로 채택했습니다.
    - **적용 효과**: 복잡한 하이퍼파라미터 튜닝에 매몰되지 않고도 모델 자체의 높은 강인함 덕분에 안정적인 예측 재현성을 확보했습니다.
  - **Time-based Validation (시계열 분리 검증)**
    - **채택 이유**: 부동산 가격 예측은 기본적으로 미래 시점을 맞추는 시계열 과제이므로, 무작위 K-Fold 방식을 쓰면 과거와 미래가 섞여 심각한 데이터 누수가 발생하기 때문입니다.
    - **적용 효과**: 실제 대회 평가 방식과 완벽히 일치하는 검증 환경을 구축함으로써 모델의 일반화 성능을 높여 Private RMSE 36,876.34를 기록했습니다.

---

### 06. IR 경진대회 — 한국어 과학지식 RAG 검색 시스템
> **IR_3조 · 2026.05 ~ 2026.06 (2주) · MAP@3 평가**
> **담당**: Hybrid 검색(BM25+Dense) 및 RRF 융합 최적화, 쿼리 다각화, Reranker 비교
> **기술 스택**: Elasticsearch, BGE-M3, RRF 앙상블, Cross-Encoder Reranking, Step-Back Prompting, Function Calling

- **파이프라인 및 모델 채택 이유 & 효과**:
  - **BGE-M3 & Elasticsearch (Hybrid + RRF)**
    - **채택 이유**: 키워드 매칭 위주의 단어 검색(BM25)과 의미 기반의 밀집 벡터 검색(Dense)은 각각 장단점이 뚜렷합니다. 두 방식의 장점을 모두 취하고 순위 역전을 방지하기 위해 RRF 알고리즘으로 융합하는 파이프라인을 설계했습니다.
    - **적용 효과**: 전문적인 과학 지식의 고유 명사와 맥락적 의미를 놓치지 않고 안정적으로 상위 후보 문서를 건져 올릴 수 있었습니다.
  - **Cross-Encoder Reranking (`max_len=2048`)**
    - **채택 이유**: 1차로 추려낸 후보 문서들 중에서 실제 질의와 가장 일치하는 정답 문맥을 정밀하게 재평가해 최종 상위권 검색 정확도(MAP@3)를 극대화하고자 도입했습니다.
    - **적용 효과**: 79회의 치열한 실험 끝에 베이스라인 대비 비약적으로 성능을 끌어올려 최고점인 MAP@3 0.9409를 달성했습니다.

---

## 📬 Contact
- **Email**: jinny040403@gmail.com
- **GitHub**: [github.com/Eojinhee](https://github.com/Eojinhee)
