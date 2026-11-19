"""
JailbreaksOverTime 데이터셋 원클릭 다운로드 스크립트 (v2)

사용 방법: VSCode에서 이 파일을 열고 우측 상단 실행(▶) 버튼을 누르거나
          터미널에서 `python download_jailbreaks_over_time.py` 를 실행합니다.

수행 내용
1. 필요한 패키지(pandas) 확인 및 자동 설치
2. 데이터셋 파일 목록 조회 후 data/JailbreaksOverTime/ 에 다운로드
   - 압축 전송을 사용하지 않는 표준 라이브러리(urllib)로 받으므로
     huggingface_hub / httpx 의 압축 해제 오류(output_buffer_limit)의 영향을 받지 않습니다.
3. 받은 파일을 pandas 로 읽어 CSV로 저장하고 기본 정보를 출력

데이터셋: https://huggingface.co/datasets/djapp18/JailbreaksOverTime
"""

import importlib
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ID = "djapp18/JailbreaksOverTime"
REVISION = "main"
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "JailbreaksOverTime"
CSV_PATH = BASE_DIR / "data" / "JailbreaksOverTime_train.csv"
HF = "https://huggingface.co"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"}


def ensure_packages() -> None:
    try:
        importlib.import_module("pandas")
        print("[1/3] 필요한 패키지가 이미 설치되어 있습니다.")
    except ImportError:
        print("[1/3] pandas 설치 중")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pandas"])


def _open(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60)


def list_files() -> list[str]:
    url = f"{HF}/api/datasets/{REPO_ID}/tree/{REVISION}?recursive=true"
    with _open(url) as r:
        items = json.loads(r.read().decode("utf-8"))
    return [it["path"] for it in items if it.get("type") == "file"]


def download_raw() -> None:
    print(f"[2/3] 원본 파일 다운로드 중 -> {RAW_DIR}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for rel in list_files():
        dest = RAW_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF}/datasets/{REPO_ID}/resolve/{REVISION}/{urllib.parse.quote(rel)}"
        with _open(url) as r, open(dest, "wb") as f:
            while chunk := r.read(1024 * 1024):
                f.write(chunk)
        print(f"    - {rel} ({dest.stat().st_size / 1024 / 1024:.2f} MB)")


def load_dataframe():
    import pandas as pd

    readers = {
        ".parquet": pd.read_parquet,
        ".csv": pd.read_csv,
    }
    candidates = [
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".parquet", ".csv"}
    ]
    if not candidates:
        raise FileNotFoundError("데이터 파일(json/jsonl/parquet/csv)을 찾지 못했습니다.")
    # 가장 큰 파일을 본 데이터로 간주
    target = max(candidates, key=lambda p: p.stat().st_size)
    print(f"    읽는 파일: {target.relative_to(RAW_DIR)}")
    suffix = target.suffix.lower()
    if suffix in readers:
        return readers[suffix](target)
    try:
        return pd.read_json(target, lines=True)
    except ValueError:
        return pd.read_json(target)


def export_csv() -> None:
    print("[3/3] 데이터 로드 및 CSV 저장 중")
    df = load_dataframe()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"    CSV 저장 완료: {CSV_PATH}")
    print(f"    행 수: {len(df):,}")
    print(f"    컬럼: {list(df.columns)}")
    if "label" in df.columns:
        print("    라벨 분포 (1=탈옥, 0=정상):")
        print(df["label"].value_counts().to_string())
    if "timestamp" in df.columns:
        print(f"    기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")


def main() -> None:
    try:
        ensure_packages()
        download_raw()
        export_csv()
        print("\n모든 작업이 완료되었습니다.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[오류] {type(exc).__name__}: {exc}")
        print("인터넷 연결 및 파이썬 환경(VSCode 우측 하단 인터프리터)을 확인해 주십시오.")
        raise


if __name__ == "__main__":
    main()
