#!/usr/bin/env bash
# Solana contract 本地测试脚本（surfpool）
#
# 环境依赖（均已就绪，无需手动配置）：
#   1. 端口：本机 WSL2 mirrored 模式下 8000-15000 被宿主保留，RPC 用 18999
#   2. 代理：surfpool 在线模式需访问 mainnet 懒加载 mpl 程序，
#      直接继承 shell 的全局代理（HTTP_PROXY/HTTPS_PROXY），无需在此设置
#
# 用法：
#   ./run-tests.sh            # 全部流程：起 surfpool -> 部署 -> 跑测试 -> 结束
set -euo pipefail
cd "$(dirname "$0")"

RPC_PORT=${RPC_PORT:-18999}
WS_PORT=${WS_PORT:-19000}

# 脚本结束（含失败/Ctrl+C）时杀掉自启动的 surfpool，不残留后台进程
cleanup() {
  pkill -9 -x surfpool 2>/dev/null || true
}
trap cleanup EXIT

# 清理旧实例，确保全新 ledger
pkill -9 -x surfpool 2>/dev/null || true
sleep 1

# 写入 surfpool IaC 配置（本地环境：18999 + 作弊码秒部署）
mkdir -p runbooks/deployment
cat > txtx.yml <<'EOF'
---
name: solana
id: solana
runbooks:
  - name: deployment
    description: Deploy programs
    location: runbooks/deployment
environments:
  localnet:
      network_id: localnet
      rpc_api_url: http://127.0.0.1:18999
  devnet:
      network_id: devnet
      rpc_api_url: https://api.devnet.solana.com
      payer_keypair_json: ~/.config/solana/id.json
      authority_keypair_json: ~/.config/solana/id.json
EOF
cat > runbooks/deployment/main.tx <<'EOF'
################################################################
# Manage solana deployment through Crypto Infrastructure as Code
################################################################

addon "svm" {
    rpc_api_url = input.rpc_api_url
    network_id = input.network_id
}

action "deploy_solana_contract" "svm::deploy_program" {
    description = "Deploy solana_contract program"
    program = svm::get_program_from_anchor_project("solana_contract")
    authority = signer.authority
    payer = signer.payer
    // 本地 surfnet 用作弊码秒部署（不用真实部署交易，绕开 RPC URL 问题）
    instant_surfnet_deployment = true
}
EOF
cat > runbooks/deployment/signers.localnet.tx <<'EOF'
signer "authority" "svm::secret_key" {
    description = "Can upgrade programs and manage critical ops"
    keypair_json = "~/.config/solana/id.json"
}

signer "payer" "svm::secret_key" {
    description = "Pays fees for program deployments"
    keypair_json = "~/.config/solana/id.json"
}
EOF

echo "== Starting surfpool (online, fresh ledger) =="
setsid env NO_DNA=1 surfpool start --network mainnet \
  --port "$RPC_PORT" --ws-port "$WS_PORT" \
  --no-tui --no-studio --yes \
  > /tmp/sp-run.log 2>&1 < /dev/null &
disown

# 等待 runbook 部署完成
for i in $(seq 1 60); do
  if grep -q "Runbook 'deployment' execution completed" /tmp/sp-run.log; then
    echo "== Program deployed after ${i}s =="
    break
  fi
  if grep -qE "aborted|error at runbooks" /tmp/sp-run.log; then
    echo "FAILED: runbook error"; tail -20 /tmp/sp-run.log; exit 1;
  fi
  sleep 1
done
grep -q "Runbook 'deployment' execution completed" /tmp/sp-run.log || {
  echo "FAILED: surfpool deploy did not complete"; tail -20 /tmp/sp-run.log; exit 1;
}

echo "== Running tests =="
env ANCHOR_PROVIDER_URL="http://127.0.0.1:${RPC_PORT}" NO_DNA=1 \
  anchor test --skip-local-validator --skip-deploy
