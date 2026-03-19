package com.metawebthree.payment.application;
 
import com.metawebthree.common.annotations.LogMethod;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.Map;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class CryptoWalletServiceImpl {

    @Value("${payment.crypto.wallet.hot-wallet.btc}")
    private String btcHotWallet;

    @Value("${payment.crypto.wallet.hot-wallet.eth}")
    private String ethHotWallet;

    @Value("${payment.crypto.wallet.hot-wallet.usdt}")
    private String usdtHotWallet;

    /**
     * @TODO: Invoke blockchain API or wallet SDK to transfer crypto, sign,
     *        broadcast and error
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
     * @TODO: Invoke blockchain API or wallet SDK to get balance
     */
    @LogMethod
    public BigDecimal getWalletBalance(String cryptoCurrency) {
        return switch (cryptoCurrency) {
            case "BTC" -> new BigDecimal("10.5");
            case "ETH" -> new BigDecimal("100.0");
            case "USDT" -> new BigDecimal("10000.0");
            case "USDC" -> new BigDecimal("10000.0");
            default -> BigDecimal.ZERO;
        };
    }

    /**
     * @TODO: Invoke blockchain API or wallet SDK to get balance
     */
    private String executeTransfer(ExchangeOrder order) {
        return "0x" + UUID.randomUUID().toString().replace("-", "") +
                System.currentTimeMillis();
    }

    private String getHotWalletAddress(String cryptoCurrency) {
        return switch (cryptoCurrency) {
            case "BTC" -> btcHotWallet;
            case "ETH" -> ethHotWallet;
            case "USDT" -> usdtHotWallet;
            default -> throw new RuntimeException("Unsupported crypto currency: " + cryptoCurrency);
        };
    }

    /**
     * @TODO: Invoke blockchain API or wallet SDK to get balance
     */
    @LogMethod
    public boolean verifyTransaction(String txHash, String cryptoCurrency) {
        return true;
    }

    /**
     * @TODO: Invoke blockchain API or wallet SDK to get balance
     */
    @LogMethod
    public Object getTransactionDetails(String txHash, String cryptoCurrency) {
        return Map.of(
                "txHash", txHash,
                "status", "confirmed",
                "confirmations", 6,
                "blockNumber", 12345678,
                "timestamp", System.currentTimeMillis());
    }

    /**
     * @TODO: Invoke blockchain API or wallet SDK to get balance
     */
    @LogMethod
    public String createWalletAddress(String cryptoCurrency) {
        return switch (cryptoCurrency) {
            case "BTC" -> "1" + UUID.randomUUID().toString().substring(0, 33);
            case "ETH", "USDT", "USDC" -> "0x" + UUID.randomUUID().toString().replace("-", "");
            default -> throw new RuntimeException("Unsupported crypto currency: " + cryptoCurrency);
        };
    }

    public boolean isValidAddress(String address, String cryptoCurrency) {
        return switch (cryptoCurrency) {
            case "BTC" -> address.matches("^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$");
            case "ETH", "USDT", "USDC" -> address.matches("^0x[a-fA-F0-9]{40}$");
            default -> false;
        };
    }
}
