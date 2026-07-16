package com.metawebthree.wallet.infrastructure.persistence.repository;

import com.metawebthree.wallet.domain.entity.Wallet;
import com.metawebthree.wallet.domain.repository.WalletRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class WalletRepositoryImpl implements WalletRepository {

    private final WalletMapper walletMapper;

    public WalletRepositoryImpl(WalletMapper walletMapper) {
        this.walletMapper = walletMapper;
    }

    @Override
    public Optional<Wallet> findById(Long id) {
        return Optional.ofNullable(walletMapper.selectById(id));
    }

    @Override
    public Optional<Wallet> findByUserIdAndChainType(String userId, String chainType) {
        return walletMapper.findByUserIdAndChainType(userId, chainType);
    }

    @Override
    public List<Wallet> findAll() {
        return walletMapper.selectList(null);
    }

    @Override
    public Wallet save(Wallet wallet) {
        if (wallet.getId() == null) {
            walletMapper.insert(wallet);
        } else {
            walletMapper.updateById(wallet);
        }
        return wallet;
    }

    @Override
    public void delete(Wallet wallet) {
        if (wallet != null && wallet.getId() != null) {
            walletMapper.deleteById(wallet.getId());
        }
    }
}
