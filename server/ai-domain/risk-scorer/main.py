import signal
import threading
from dotenv import load_dotenv
from pathlib import Path
import sys
from fastapi import FastAPI, Request
import uvicorn
import os
from app.risk_scorer_grpc_service import RiskScorerGrpcService
from RiskScorerService_pb2 import (
    TestRequest, TestResponse,
    ScoreRequest, ScoreResponse
)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.resolve()))

app = FastAPI()
risk_scorer_service = RiskScorerGrpcService()

def main():
    rpc()
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT", 8000))


@app.get("/test")
def test():
    return risk_scorer_service.test(TestRequest(), None).result


def rpc():
    from grpcClient import start_risk_score_model
    print("Start risk score model")
    server = start_risk_score_model()
    print(f"Already start risk score model on port {server._service._port}")
    keep_alive()


def keep_alive():
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda s, f: stop.set())
    signal.signal(signal.SIGTERM, lambda s, f: stop.set())
    return stop.wait()


if __name__ == "__main__":
    main()
