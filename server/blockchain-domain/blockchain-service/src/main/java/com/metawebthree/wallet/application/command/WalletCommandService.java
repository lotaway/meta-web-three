package com.metawebthree.wallet.application.command;

import com.metawebthree.wallet.application.dto.WalletDTO;
import com.metawebthree.wallet.domain.entity.Wallet;
import com.metawebthree.wallet.domain.repository.WalletRepository;
import com.metawebthree.wallet.domain.service.WalletDomainService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class WalletCommandService {
    private final WalletRepository walletRepository;
    private final WalletDomainService walletDomainService;

    public WalletCommandService(WalletRepository walletRepository, WalletDomainService walletDomainService) {
        this.walletRepository = walletRepository;
        this.walletDomainService = walletDomainService;
    }

    @Transactional
    public WalletDTO createWallet(String userId, String chainType, String address) {
        walletRepository.findByUserIdAndChainType(userId, chainType)
            .ifPresent(w -> { throw new IllegalStateException("Wallet already exists for this chain"); });

        Wallet wallet = walletDomainService.createWallet(userId, chainType, address);
        Wallet saved = walletRepository.save(wallet);
        return toDTO(saved);
    }

    @Transactional
    public WalletDTO deposit(Long walletId, java.math.BigDecimal amount) {
        Wallet wallet = walletRepository.findById(walletId)
            .orElseThrow(() -> new IllegalArgumentException("Wallet not found"));
        walletDomainService.deposit(wallet, amount);
        return toDTO(walletRepository.save(wallet));
    }

    @Transactional
    public WalletDTO withdraw(Long walletId, java.math.BigDecimal amount) {
        Wallet wallet = walletRepository.findById(walletId)
            .orElseThrow(() -> new IllegalArgumentException("Wallet not found"));
        walletDomainService.withdraw(wallet, amount);
        return toDTO(walletRepository.save(wallet));
    }

    private WalletDTO toDTO(Wallet wallet) {
        return new WalletDTO(wallet.getId(), wallet.getUserId(), wallet.getChainType(),
            wallet.getAddress(), wallet.getBalance(), wallet.getStatus());
    }
}