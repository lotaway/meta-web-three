package com.metawebthree.promotion.infrastructure.gateway;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.metawebthree.promotion.domain.ports.BlockchainService;

import java.math.BigDecimal;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class BlockchainServiceStubImpl implements BlockchainService {
    private static final Logger log = LoggerFactory.getLogger(BlockchainServiceStubImpl.class);

    private final Map<String, String> couponBatchRoots = new ConcurrentHashMap<>();
    private final Map<String, Map<String, Object>> wallets = new ConcurrentHashMap<>();
    private final Map<String, BigDecimal> balances = new ConcurrentHashMap<>();

    private static final BigDecimal INITIAL_BALANCE = new BigDecimal("10000");

    @Override
    public void setCouponBatchRoot(String batchId, String merkleRoot) {
        couponBatchRoots.put(batchId, merkleRoot);
        log.info("Stored coupon batch root: batchId={}, merkleRoot={}", batchId, merkleRoot);
    }

    @Override
    public String createWallet() {
        String address = "0x" + UUID.randomUUID().toString().replace("-", "");
        Map<String, Object> wallet = new ConcurrentHashMap<>();
        wallet.put("address", address);
        wallet.put("createdAt", System.currentTimeMillis());
        wallets.put(address, wallet);
        balances.put(address, INITIAL_BALANCE);
        log.info("Created stub wallet: address={}, initialBalance={}", address, INITIAL_BALANCE);
        return address;
    }

    @Override
    public BigDecimal getBalance(String address) {
        BigDecimal balance = balances.get(address);
        if (balance == null) {
            log.warn("Wallet not found: address={}", address);
            return BigDecimal.ZERO;
        }
        return balance;
    }

    @Override
    public boolean transfer(String from, String to, BigDecimal amount) {
        BigDecimal fromBalance = balances.get(from);
        BigDecimal toBalance = balances.get(to);

        if (fromBalance == null) {
            log.error("Transfer failed: source wallet not found: {}", from);
            return false;
        }
        if (toBalance == null) {
            log.error("Transfer failed: destination wallet not found: {}", to);
            return false;
        }
        if (fromBalance.compareTo(amount) < 0) {
            log.error("Transfer failed: insufficient balance. from={}, balance={}, amount={}",
                    from, fromBalance, amount);
            return false;
        }

        balances.put(from, fromBalance.subtract(amount));
        balances.put(to, toBalance.add(amount));
        log.info("Transfer successful: from={}, to={}, amount={}", from, to, amount);
        return true;
    }
}
