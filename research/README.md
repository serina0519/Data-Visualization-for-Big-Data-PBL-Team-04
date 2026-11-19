# JailbreaksOverTime — 탈옥 탐지기의 지속 적응 방식 비교 (재현 패키지)

본 저장소는 JailbreaksOverTime 벤치마크(Piet et al., 2025, AISec) 위에서
탈옥 탐지기의 갱신 규칙(비적응 / 전체 라벨 / 자기학습 / kNN 메모리 / 능동 라벨 획득 / 제안 R1)을
시간순 test-then-train 방식으로 비교한 실험 전체를 담고 있습니다.
결과 보고서는 `REPORT.md`, 숫자 표는 `outputs/metrics/`, 통계 검정은 `outputs/statistics/`,
그림은 `outputs/figures/`에 있습니다.

## 구조

```
jailbreak_research/
├── main.py                  # 전체 파이프라인 실행기 (데이터 검증 → 임베딩 → 실험 → 평가 → 통계 → 그림)
├── requirements.txt
├── README.md / REPORT.md
├── configs/version1.json, version2.json   # 설정 격자 명세 (문서용; 코드의 grid()와 동일)
├── src/
│   ├── data.py              # 입력 검증, 7일 주차 생성, QC, 초기 4주만으로 적합하는 고정 임베딩
│   ├── models.py            # 가중 L2 로지스틱 회귀(자체 구현), 코사인 kNN 메모리(B4)
│   ├── experiment.py        # 설정 격자, 라벨 접근 차단(LabelOracle), 시간순 학습 루프, R1 및 절제
│   ├── evaluation.py        # 주차별·통합·층화 지표, 임계값 무관 보조 지표
│   ├── statistics.py        # 짝지은 블록 부트스트랩, Holm 보정, 블록 길이 민감도
│   └── visualization.py     # 300 DPI PNG + 벡터 PDF 그림 6종
├── tests/                   # 누수 방지·갱신 시점·예산 정확성 단위 테스트 (pytest)
├── outputs/                 # 주 분석 (단어 1-2gram TF-IDF→SVD 96차원 임베딩)
│   ├── qc/                  # 데이터 품질 요약 (embeddings.npy 등 대용량 파일은 ZIP에서 제외)
│   ├── predictions/         # 설정×시드별 고정 예측 점수 (row_id, week, score)
│   ├── acquisition/         # 참라벨 공개 내역 (어느 행의 라벨이 몇 주차에 어떤 이유로 공개됐는지)
│   ├── metrics/             # weekly_metrics, run_summary, setting_summary, stratified_metrics, update_audit, main_table
│   ├── statistics/          # paired_block_bootstrap(.csv), embedding_sensitivity_word_vs_char.csv
│   └── figures/
└── outputs_char/            # 민감도 분석 (문자 3-5gram TF-IDF→SVD 임베딩; V1+V2 본 실험만, 예측 파일 제외)
```

## 실행 방법

```bash
pip install -r requirements.txt
python -m pytest tests -q                       # 13개 테스트
python main.py --data path/to/JailbreaksOverTime_train.csv --out outputs --workers 2
python main.py --data path/to/JailbreaksOverTime_train.csv --out outputs_char --embedding char --versions main
```

2코어 기준 전체 250회 실행에 약 3분, 통계·그림 포함 약 3.5분이 걸립니다.
`--skip-runs`를 주면 디스크의 지표를 재사용해 통계와 그림만 다시 만듭니다.

## 실험 규약 요약

- 주차: 최초 관측일 2023-02-28 00:00 UTC부터 7일 단위, 총 43주. 초기 학습 1–4주(2,050행), 평가 5–43주(21,508행).
- 임베딩은 초기 4주 텍스트로만 적합한 뒤 고정합니다. 출처(source)는 입력 특징으로 쓰지 않습니다.
- 매주 해당 주의 모든 프롬프트를 갱신 전 모델로 먼저 평가하고, 완전한 주기가 끝난 시점에만 갱신합니다. 마지막 불완전 주기는 평가만 합니다.
- 판정 임계값은 모든 방법에서 0.5로 고정합니다. 보조로 AUROC와 FPR 1%·5%에서의 Recall을 함께 보고합니다.
- 참라벨은 `LabelOracle.reveal()`을 통해서만 얻을 수 있으며, 비지도 방법(A1, B1–B4)이 초기 4주 밖의 라벨을 요청하면 즉시 실패합니다.
- C1·C2·R1의 라벨 획득 개수는 ⌊k·N_cycle⌋이며 최소 1개 규칙은 없습니다. 실제 라벨 비율은 명목치보다 약간 작습니다(예: 10% → 9.4–9.9%).
- C1과 R1(및 절제)은 시드 5개(0–4)로 반복합니다. 나머지는 결정적이므로 1회입니다.
- 통계: 주차별 혼동행렬을 방법 간 짝지은 뒤, 길이 2/4/8주의 순환 이동블록 부트스트랩(2,000회)으로 Recall·FPR·F1 차이와 95% 신뢰구간, 양측 p값을 구하고 비교 가족별로 Holm 보정을 적용합니다.
