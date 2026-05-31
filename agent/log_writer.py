"""
agent/log_writer.py — 터미널 로그 자동 저장

역할:
    sys.stdout을 TeeWriter로 교체하여 모든 print() 출력이
    콘솔과 로그 파일에 동시에 기록되도록 한다.
    기존 코드 변경 없이 agent_config.yaml의 logging 설정만으로 제어한다.

데이터 흐름:
    입력: config (agent_config.yaml의 logging 섹션), session_id
    출력: logs/{timestamp}_{session_id}.log 파일
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

# --- 모듈 레벨 상태 ---
_original_stdout: TextIO | None = None    # 복원용 원래 stdout
_log_file: TextIO | None = None           # 열린 로그 파일 핸들


class TeeWriter:
    """stdout을 감싸서 콘솔 + 파일에 동시 기록하는 Writer.

    sys.stdout을 이 객체로 교체하면 모든 print() 호출이
    원래 콘솔과 로그 파일 양쪽에 기록된다.
    """

    def __init__(self, console: TextIO, log_file: TextIO) -> None:
        """TeeWriter를 초기화한다.

        Args:
            console: 원래 sys.stdout (터미널 출력용).
            log_file: 로그 파일 핸들 (파일 기록용).
        """
        self.console = console
        self.log_file = log_file

    def write(self, text: str) -> int:
        """텍스트를 콘솔과 파일에 동시에 기록한다.

        Args:
            text: 기록할 텍스트.

        Returns:
            기록된 문자 수.
        """
        self.console.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        """콘솔과 파일 버퍼를 모두 비운다."""
        self.console.flush()
        self.log_file.flush()

    # print()가 내부적으로 확인하는 속성들을 원래 stdout에서 위임한다
    @property
    def encoding(self) -> str:
        """콘솔의 인코딩을 반환한다."""
        return self.console.encoding

    def isatty(self) -> bool:
        """원래 콘솔의 isatty 결과를 반환한다."""
        return self.console.isatty()

    def fileno(self) -> int:
        """원래 콘솔의 파일 디스크립터를 반환한다."""
        return self.console.fileno()


def setup_session_log(config: dict, session_id: str) -> None:
    """세션 시작 시 로그 파일을 생성하고 TeeWriter를 설치한다.

    config의 logging.enabled가 False이면 아무 동작도 하지 않는다.
    파일명 형식: {YYYYMMDD_HHMMSS}_{session_id}.log

    Args:
        config: agent_config.yaml에서 로드한 설정 딕셔너리.
        session_id: 세션 식별자 (예: "sess_a1b2c3d4").
    """
    global _original_stdout, _log_file

    logging_config = config.get("logging", {})
    if not logging_config.get("enabled", False):
        return

    log_dir = Path(logging_config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{session_id}.log"

    _original_stdout = sys.stdout
    _log_file = open(log_path, "w", encoding="utf-8")

    sys.stdout = TeeWriter(_original_stdout, _log_file)  # type: ignore[assignment]

    print(f"[로그 저장] {log_path}")


def teardown_session_log() -> None:
    """세션 종료 시 원래 stdout을 복원하고 로그 파일을 닫는다."""
    global _original_stdout, _log_file

    if _original_stdout is None:
        return

    sys.stdout = _original_stdout
    _original_stdout = None

    if _log_file is not None:
        _log_file.close()
        _log_file = None
