package com.metawebthree.wallet.domain.service;

import com.metawebthree.wallet.domain.entity.Wallet;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;

@Service
public class WalletDomainServiceImpl implements WalletDomainService {

    @Override
    public Wallet createWallet(String userId, String chainType, String address) {
        return new Wallet(userId, chainType, address);
    }

    @Override
    public void deposit(Wallet wallet, BigDecimal amount) {
        wallet.deposit(amount);
    }

    @Override
    public void withdraw(Wallet wallet, BigDecimal amount) {
        if (!hasEnoughBalance(wallet, amount)) {
            throw new IllegalStateException("Insufficient balance");
        }
        wallet.withdraw(amount);
    }

    @Override
    public boolean hasEnoughBalance(Wallet wallet, BigDecimal amount) {
        return wallet.getBalance().compareTo(amount) >= 0;
    }
}