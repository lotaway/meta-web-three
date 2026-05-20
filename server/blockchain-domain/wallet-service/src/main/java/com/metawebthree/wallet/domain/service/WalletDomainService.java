package com.metawebthree.wallet.domain.service;

import com.metawebthree.wallet.domain.entity.Wallet;
import java.math.BigDecimal;

public interface WalletDomainService {
    Wallet createWallet(String userId, String chainType, String address);
    void deposit(Wallet wallet, BigDecimal amount);
    void withdraw(Wallet wallet, BigDecimal amount);
    boolean hasEnoughBalance(Wallet wallet, BigDecimal amount);
}