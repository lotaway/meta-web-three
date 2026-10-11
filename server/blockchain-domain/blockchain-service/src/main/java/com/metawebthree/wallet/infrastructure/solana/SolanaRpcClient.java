package com.metawebthree.wallet.infrastructure.solana;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.math.BigInteger;
import java.util.List;
import java.util.Map;

@Component
public class SolanaRpcClient {

    private final RestTemplate restTemplate;
    private final String rpcUrl;

    public SolanaRpcClient(@Value("${blockchain.solana.rpc-url}") String rpcUrl) {
        this.restTemplate = new RestTemplate();
        this.rpcUrl = rpcUrl;
    }

    public BigInteger getBalance(String pubkey) {
        Map<String, Object> result = call("getBalance", List.of(pubkey));
        return new BigInteger(result.get("value").toString());
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getAccountInfo(String pubkey) {
        return call("getAccountInfo", List.of(pubkey, Map.of("encoding", "jsonParsed")));
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getTokenSupply(String mint) {
        return call("getTokenSupply", List.of(mint));
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getTokenAccountBalance(String tokenAccount) {
        return call("getTokenAccountBalance", List.of(tokenAccount));
    }

    @SuppressWarnings("unchecked")
    public String sendTransaction(String signedTransaction) {
        Map<String, Object> result = call("sendTransaction", List.of(signedTransaction, Map.of("encoding", "base64")));
        return (String) result.get("signature");
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getTransaction(String signature) {
        return call("getTransaction", List.of(signature, Map.of("encoding", "jsonParsed", "maxSupportedTransactionVersion", 0)));
    }

    @SuppressWarnings("unchecked")
    public long getSlot() {
        Map<String, Object> result = call("getSlot", List.of());
        return ((Number) result.get("slot")).longValue();
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getLatestBlockhash() {
        Map<String, Object> result = call("getLatestBlockhash", List.of());
        return (Map<String, Object>) result.get("value");
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> call(String method, List<Object> params) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        Map<String, Object> body = Map.of(
            "jsonrpc", "2.0",
            "id", 1,
            "method", method,
            "params", params
        );
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);
        Map<String, Object> response = restTemplate.postForObject(rpcUrl, request, Map.class);
        if (response != null && response.containsKey("error")) {
            Map<String, Object> error = (Map<String, Object>) response.get("error");
            throw new RuntimeException("Solana RPC error: " + error.get("message"));
        }
        return response != null ? (Map<String, Object>) response.get("result") : Map.of();
    }
}
