package com.metawebthree.payment.application;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.common.annotations.LogMethod;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Service
@RequiredArgsConstructor
@Slf4j
public class CryptoWalletServiceImpl {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${payment.crypto.wallet.hot-wallet.btc}")
    private String btcHotWallet;

    @Value("${payment.crypto.wallet.hot-wallet.eth}")
    private String ethHotWallet;

    @Value("${payment.crypto.wallet.hot-wallet.usdt}")
    private String usdtHotWallet;

    @Value("${payment.crypto.wallet.api.blockchain-url:}")
    private String blockchainApiUrl;

    @Value("${payment.crypto.wallet.api.wallet-sdk-url:}")
    private String walletSdkUrl;

    @Value("${payment.crypto.wallet.api.key:}")
    private String apiKey;

    // Cache for wallet addresses to avoid generating duplicates
    private final Map<String, String> addressCache = new ConcurrentHashMap<>();

    /**
     * Transfer crypto currency to specified address
     * Invokes blockchain API or wallet SDK to transfer, sign, broadcast
     */
    @LogMethod
    public String transferCrypto(ExchangeOrder order) {
        validateWalletBalance(order);
        String txHash = executeTransfer(order);
        return txHash;
    }

    private void validateWalletBalance(ExchangeOrder order) {
        BigDecimal balance = getWalletBalance(order.getCryptoCurrency());
        if (balance.compareTo(order.getCryptoAmount()) < 0) {
            throw new RuntimeException("Insufficient wallet balance. Required: " + order.getCryptoAmount() + " "
                    + order.getCryptoCurrency() +
                    ", Available: " + balance + " " + order.getCryptoCurrency());
        }
    }

    /**
     * Get wallet balance from blockchain API or wallet SDK
     */
    @LogMethod
    public BigDecimal getWalletBalance(String cryptoCurrency) {
        // Try blockchain API first
        if (blockchainApiUrl != null && !blockchainApiUrl.isEmpty()) {
            try {
                return fetchBalanceFromBlockchainApi(cryptoCurrency);
            } catch (Exception e) {
                log.warn("Failed to fetch balance from blockchain API: {}, falling back to wallet SDK", e.getMessage());
            }
        }
        // Fallback to wallet SDK
        if (walletSdkUrl != null && !walletSdkUrl.isEmpty()) {
            try {
                return fetchBalanceFromWalletSdk(cryptoCurrency);
            } catch (Exception e) {
                log.error("Failed to fetch balance from wallet SDK: {}", e.getMessage());
            }
        }
        // No API configured, return zero
        log.warn("No blockchain API or wallet SDK configured, returning zero balance");
        return BigDecimal.ZERO;
    }

    private BigDecimal fetchBalanceFromBlockchainApi(String cryptoCurrency) {
        String address = getHotWalletAddress(cryptoCurrency);
        String url = blockchainApiUrl + "/balance?address=" + address + "&currency=" + cryptoCurrency;
        
        HttpHeaders headers = new HttpHeaders();
        if (apiKey != null && !apiKey.isEmpty()) {
            headers.set("X-API-Key", apiKey);
        }
        HttpEntity<String> entity = new HttpEntity<>(headers);
        
        ResponseEntity<Map> response = restTemplate.exchange(url, HttpMethod.GET, entity, Map.class);
        
        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            Object balanceObj = response.getBody().get("balance");
            if (balanceObj != null) {
                return new BigDecimal(balanceObj.toString());
            }
        }
        throw new RuntimeException("Failed to fetch balance from blockchain API");
    }

    private BigDecimal fetchBalanceFromWalletSdk(String cryptoCurrency) {
        String address = getHotWalletAddress(cryptoCurrency);
        String url = walletSdkUrl + "/wallet/balance";
        
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("address", address);
        requestBody.put("currency", cryptoCurrency);
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        if (apiKey != null && !apiKey.isEmpty()) {
            headers.set("X-API-Key", apiKey);
        }
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);
        
        ResponseEntity<Map> response = restTemplate.postForEntity(url, entity, Map.class);
        
        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            Object balanceObj = response.getBody().get("balance");
            if (balanceObj != null) {
                return new BigDecimal(balanceObj.toString());
            }
        }
        throw new RuntimeException("Failed to fetch balance from wallet SDK");
    }

    /**
     * Execute transfer via blockchain API or wallet SDK
     */
    private String executeTransfer(ExchangeOrder order) {
        if ((blockchainApiUrl == null || blockchainApiUrl.isEmpty()) && 
            (walletSdkUrl == null || walletSdkUrl.isEmpty())) {
            log.error("No blockchain API or wallet SDK configured for transfer");
            return null;
        }
        
        if (blockchainApiUrl != null && !blockchainApiUrl.isEmpty()) {
            try {
                return transferViaBlockchainApi(order);
            } catch (Exception e) {
                log.warn("Failed to transfer via blockchain API: {}, falling back to wallet SDK", e.getMessage());
            }
        }
        if (walletSdkUrl != null && !walletSdkUrl.isEmpty()) {
            try {
                return transferViaWalletSdk(order);
            } catch (Exception e) {
                log.error("Failed to transfer via wallet SDK: {}", e.getMessage());
                return null;
            }
        }
        return null;
    }

    private String transferViaBlockchainApi(ExchangeOrder order) {
        String fromAddress = getHotWalletAddress(order.getCryptoCurrency());
        String toAddress = order.getUserWalletAddress();
        
        String url = blockchainApiUrl + "/transfer";
        
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("from", fromAddress);
        requestBody.put("to", toAddress);
        requestBody.put("amount", order.getCryptoAmount().toPlainString());
        requestBody.put("currency", order.getCryptoCurrency());
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        if (apiKey != null && !apiKey.isEmpty()) {
            headers.set("X-API-Key", apiKey);
        }
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);
        
        ResponseEntity<Map> response = restTemplate.postForEntity(url, entity, Map.class);
        
        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            Object txHashObj = response.getBody().get("txHash");
            if (txHashObj != null) {
                return txHashObj.toString();
            }
        }
        throw new RuntimeException("Transfer failed: no txHash returned");
    }

    private String transferViaWalletSdk(ExchangeOrder order) {
        String fromAddress = getHotWalletAddress(order.getCryptoCurrency());
        String toAddress = order.getUserWalletAddress();
        
        String url = walletSdkUrl + "/wallet/transfer";
        
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("fromAddress", fromAddress);
        requestBody.put("toAddress", toAddress);
        requestBody.put("amount", order.getCryptoAmount().toPlainString());
        requestBody.put("currency", order.getCryptoCurrency());
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        if (apiKey != null && !apiKey.isEmpty()) {
            headers.set("X-API-Key", apiKey);
        }
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);
        
        ResponseEntity<Map> response = restTemplate.postForEntity(url, entity, Map.class);
        
        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            Object txHashObj = response.getBody().get("txHash");
            if (txHashObj != null) {
                return txHashObj.toString();
            }
            Object hashObj = response.getBody().get("hash");
            if (hashObj != null) {
                return hashObj.toString();
            }
        }
        throw new RuntimeException("Transfer failed: no txHash returned");
    }

    private String getHotWalletAddress(String cryptoCurrency) {
        return switch (cryptoCurrency) {
            case "BTC" -> btcHotWallet;
            case "ETH" -> ethHotWallet;
            case "USDT" -> usdtHotWallet;
            default -> {
                log.error("Unsupported crypto currency: {}", cryptoCurrency);
                yield null;
            }
        };
    }

    /**
     * Verify transaction status via blockchain API or wallet SDK
     */
    @LogMethod
    public boolean verifyTransaction(String txHash, String cryptoCurrency) {
        // Try blockchain API first
        if (blockchainApiUrl != null && !blockchainApiUrl.isEmpty()) {
            try {
                return verifyTransactionViaBlockchainApi(txHash, cryptoCurrency);
            } catch (Exception e) {
                log.warn("Failed to verify via blockchain API: {}", e.getMessage());
            }
        }
        // Fallback to wallet SDK
        if (walletSdkUrl != null && !walletSdkUrl.isEmpty()) {
            try {
                return verifyTransactionViaWalletSdk(txHash, cryptoCurrency);
            } catch (Exception e) {
                log.error("Failed to verify via wallet SDK: {}", e.getMessage());
            }
        }
        // No API configured, cannot verify
        log.warn("No blockchain API or wallet SDK configured, cannot verify transaction");
        return false;
    }

    private boolean verifyTransactionViaBlockchainApi(String txHash, String cryptoCurrency) {
        String url = blockchainApiUrl + "/transaction/" + txHash + "?currency=" + cryptoCurrency;
        
        HttpHeaders headers = new HttpHeaders();
        if (apiKey != null && !apiKey.isEmpty()) {
            headers.set("X-API-Key", apiKey);
        }
        HttpEntity<String> entity = new HttpEntity<>(headers);
        
        ResponseEntity<Map> response = restTemplate.exchange(url, HttpMethod.GET, entity, Map.class);
        
        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            String status = (String) response.getBody().get("status");
            return "confirmed".equalsIgnoreCase(status) || "success".equalsIgnoreCase(status);
        }
        return false;
    }

    private boolean verifyTransactionViaWalletSdk(String txHash, String cryptoCurrency) {
        String url = walletSdkUrl + "/wallet/transaction/status";
        
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("txHash", txHash);
        requestBody.put("currency", cryptoCurrency);
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        if (apiKey != null && !apiKey.isEmpty()) {
            headers.set("X-API-Key", apiKey);
        }
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);
        
        ResponseEntity<Map> response = restTemplate.postForEntity(url, entity, Map.class);
        
        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            String status = (String) response.getBody().get("status");
            return "confirmed".equalsIgnoreCase(status) || "success".equalsIgnoreCase(status);
        }
        return false;
    }

    /**
     * Get transaction details via blockchain API or wallet SDK
     */
    @LogMethod
    public Object getTransactionDetails(String txHash, String cryptoCurrency) {
        // Try blockchain API first
        if (blockchainApiUrl != null && !blockchainApiUrl.isEmpty()) {
            try {
                return fetchTransactionDetailsFromBlockchainApi(txHash, cryptoCurrency);
            } catch (Exception e) {
                log.warn("Failed to fetch transaction details via blockchain API: {}", e.getMessage());
            }
        }
        // Fallback to wallet SDK
        if (walletSdkUrl != null && !walletSdkUrl.isEmpty()) {
            try {
                return fetchTransactionDetailsFromWalletSdk(txHash, cryptoCurrency);
            } catch (Exception e) {
                log.error("Failed to fetch transaction details via wallet SDK: {}", e.getMessage());
            }
        }
        // Return empty map if no API configured
        return Map.of(
                "txHash", txHash,
                "status", "unknown",
                "error", "No blockchain API or wallet SDK configured");
    }

    private Object fetchTransactionDetailsFromBlockchainApi(String txHash, String cryptoCurrency) {
        String url = blockchainApiUrl + "/transaction/" + txHash + "?currency=" + cryptoCurrency;
        
        HttpHeaders headers = new HttpHeaders();
        if (apiKey != null && !apiKey.isEmpty()) {
            headers.set("X-API-Key", apiKey);
        }
        HttpEntity<String> entity = new HttpEntity<>(headers);
        
        ResponseEntity<Map> response = restTemplate.exchange(url, HttpMethod.GET, entity, Map.class);
        
        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            return response.getBody();
        }
        throw new RuntimeException("Failed to fetch transaction details");
    }

    private Object fetchTransactionDetailsFromWalletSdk(String txHash, String cryptoCurrency) {
        String url = walletSdkUrl + "/wallet/transaction/details";
        
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("txHash", txHash);
        requestBody.put("currency", cryptoCurrency);
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        if (apiKey != null && !apiKey.isEmpty()) {
            headers.set("X-API-Key", apiKey);
        }
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);
        
        ResponseEntity<Map> response = restTemplate.postForEntity(url, entity, Map.class);
        
        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            return response.getBody();
        }
        throw new RuntimeException("Failed to fetch transaction details");
    }

    /**
     * Create new wallet address via blockchain API or wallet SDK
     */
    @LogMethod
    public String createWalletAddress(String cryptoCurrency) {
        if ((blockchainApiUrl == null || blockchainApiUrl.isEmpty()) && 
            (walletSdkUrl == null || walletSdkUrl.isEmpty())) {
            log.error("No blockchain API or wallet SDK configured for address creation");
            return null;
        }
        
        String cacheKey = "wallet:" + cryptoCurrency;
        if (addressCache.containsKey(cacheKey)) {
            return addressCache.get(cacheKey);
        }
        
        if (blockchainApiUrl != null && !blockchainApiUrl.isEmpty()) {
            try {
                String address = createWalletAddressViaBlockchainApi(cryptoCurrency);
                addressCache.put(cacheKey, address);
                return address;
            } catch (Exception e) {
                log.warn("Failed to create wallet address via blockchain API: {}", e.getMessage());
            }
        }
        if (walletSdkUrl != null && !walletSdkUrl.isEmpty()) {
            try {
                String address = createWalletAddressViaWalletSdk(cryptoCurrency);
                addressCache.put(cacheKey, address);
                return address;
            } catch (Exception e) {
                log.error("Failed to create wallet address via wallet SDK: {}", e.getMessage());
                return null;
            }
        }
        return null;
    }

    private String createWalletAddressViaBlockchainApi(String cryptoCurrency) {
        String url = blockchainApiUrl + "/wallet/create";
        
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("currency", cryptoCurrency);
        requestBody.put("type", "hot");
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        if (apiKey != null && !apiKey.isEmpty()) {
            headers.set("X-API-Key", apiKey);
        }
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);
        
        ResponseEntity<Map> response = restTemplate.postForEntity(url, entity, Map.class);
        
        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            Object addressObj = response.getBody().get("address");
            if (addressObj != null) {
                return addressObj.toString();
            }
        }
        throw new RuntimeException("Failed to create wallet address: no address returned");
    }

    private String createWalletAddressViaWalletSdk(String cryptoCurrency) {
        String url = walletSdkUrl + "/wallet/create";
        
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("currency", cryptoCurrency);
        requestBody.put("walletType", "hot");
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        if (apiKey != null && !apiKey.isEmpty()) {
            headers.set("X-API-Key", apiKey);
        }
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);
        
        ResponseEntity<Map> response = restTemplate.postForEntity(url, entity, Map.class);
        
        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            Object addressObj = response.getBody().get("address");
            if (addressObj != null) {
                return addressObj.toString();
            }
            Object addressObj2 = response.getBody().get("walletAddress");
            if (addressObj2 != null) {
                return addressObj2.toString();
            }
        }
        throw new RuntimeException("Failed to create wallet address: no address returned");
    }

    public boolean isValidAddress(String address, String cryptoCurrency) {
        return switch (cryptoCurrency) {
            case "BTC" -> address.matches("^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$");
            case "ETH", "USDT", "USDC" -> address.matches("^0x[a-fA-F0-9]{40}$");
            default -> false;
        };
    }
}
