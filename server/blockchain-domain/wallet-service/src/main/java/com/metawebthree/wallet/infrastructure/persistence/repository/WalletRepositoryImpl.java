package com.metawebthree.wallet.infrastructure.persistence.repository;

import com.metawebthree.wallet.domain.entity.Wallet;
import com.metawebthree.wallet.domain.repository.WalletRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class WalletRepositoryImpl implements WalletRepository {

    private final Map<Long, Wallet> walletMap = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);

    @Override
    public Optional<Wallet> findById(Long id) {
        return Optional.ofNullable(walletMap.get(id));
    }

    @Override
    public Optional<Wallet> findByUserIdAndChainType(String userId, String chainType) {
        return walletMap.values().stream()
                .filter(w -> w.getUserId().equals(userId) && w.getChainType().equals(chainType))
                .findFirst();
    }

    @Override
    public List<Wallet> findAll() {
        return new ArrayList<>(walletMap.values());
    }

    @Override
    public Wallet save(Wallet wallet) {
        if (wallet.getId() == null) {
            wallet.setId(idGenerator.getAndIncrement());
        }
        walletMap.put(wallet.getId(), wallet);
        return wallet;
    }

    @Override
    public void delete(Wallet wallet) {
        if (wallet != null && wallet.getId() != null) {
            walletMap.remove(wallet.getId());
        }
    }
}