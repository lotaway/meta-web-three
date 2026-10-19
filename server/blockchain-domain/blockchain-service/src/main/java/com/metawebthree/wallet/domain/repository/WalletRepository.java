package com.metawebthree.wallet.domain.repository;

import com.metawebthree.wallet.domain.entity.Wallet;
import java.util.List;
import java.util.Optional;

public interface WalletRepository {
    Optional<Wallet> findById(Long id);
    Optional<Wallet> findByUserIdAndChainType(String userId, String chainType);
    List<Wallet> findAll();
    List<Wallet> findByUserId(String userId);
    Wallet save(Wallet wallet);
    void delete(Wallet wallet);
}