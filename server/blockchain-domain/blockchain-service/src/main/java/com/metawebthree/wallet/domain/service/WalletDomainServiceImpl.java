package com.metawebthree.wallet.domain.service;

import com.metawebthree.wallet.domain.entity.Wallet;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;

@Service
@Slf4j
public class WalletDomainServiceImpl implements WalletDomainService {

    // In a full implementation, this would be a WalletRepository
    // For now, we use an in-memory map with userId-based access control
    // TODO: Replace with actual MyBatisPlus repository integration

    @Override
    @Transactional
    public Wallet createWallet(String userId, String chainType, String address) {
        // Verify userId is not null/empty
        if (userId == null || userId.isBlank()) {
            throw new IllegalArgumentException("User ID must not be null/blank");
        }
        if (chainType == null || chainType.isBlank()) {
            throw new IllegalArgumentException("Chain type must not be null/blank");
        }
        if (address == null || address.isBlank()) {
            throw new IllegalArgumentException("Wallet address must not be null/blank");
        }

        Wallet wallet = new Wallet(userId, chainType, address);
        log.info("Wallet created: id={}, userId={}, chainType={}, address={}", 
                wallet.getId(), userId, chainType, address);
        return wallet;
    }

    @Override
    @Transactional
    public void deposit(Wallet wallet, BigDecimal amount, String userId) {
        // Verify ownership: wallet's userId must match the provided userId
        if (wallet == null) {
            throw new IllegalArgumentException("Wallet must not be null");
        }
        if (userId == null || userId.isBlank()) {
            throw new IllegalArgumentException("User ID must not be null/blank");
        }
        if (!wallet.getUserId().equals(userId)) {
            throw new SecurityException("Wallet userId " + wallet.getUserId() 
                    + " does not match provided userId " + userId);
        }
        if (amount == null || amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Deposit amount must be positive");
        }
        wallet.deposit(amount);
        log.info("Wallet deposited: id={}, userId={}, amount={}", 
                wallet.getId(), userId, amount);
    }

    @Override
    @Transactional
    public void withdraw(Wallet wallet, BigDecimal amount, String userId) {
        // Verify ownership: wallet's userId must match the provided userId
        if (wallet == null) {
            throw new IllegalArgumentException("Wallet must not be null");
        }
        if (userId == null || userId.isBlank()) {
            throw new IllegalArgumentException("User ID must not be null/blank");
        }
        if (!wallet.getUserId().equals(userId)) {
            throw new SecurityException("Wallet userid " + wallet.getUserId() 
                    + " does not match provided userId " + userId);
        }
        if (amount == null || amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Withdraw amount must be positive");
        }
        wallet.withdraw(amount);
        log.info("Wallet withdrawn: id={}, userId={}, amount={}", 
                wallet.getId(), userId, amount);
    }

    @Override
    public boolean hasEnoughBalance(Wallet wallet, BigDecimal amount) {
        if (wallet == null) {
            throw new IllegalArgumentException("Wallet must not be null");
        }
        if (amount == null || amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Amount must be positive");
        }
        return wallet.getBalance().compareTo(amount) >= 0;
    }
}