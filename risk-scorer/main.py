import signal
import threading
from RiskScoreModel import start_risk_score_model
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


def main():
    print("Start risk score model in rpc")
    server = start_risk_score_model()
    print(f"Already start risk score model in rpc on port {server._service._port}")
    keep_alive()


def keep_alive():
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda s, f: stop.set())
    signal.signal(signal.SIGTERM, lambda s, f: stop.set())
    stop.wait()


if __name__ == "__main__":
    main()
