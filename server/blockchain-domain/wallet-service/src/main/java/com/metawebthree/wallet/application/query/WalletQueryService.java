package com.metawebthree.wallet.application.query;

import com.metawebthree.wallet.application.dto.WalletDTO;
import com.metawebthree.wallet.domain.entity.Wallet;
import com.metawebthree.wallet.domain.repository.WalletRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class WalletQueryService {
    private final WalletRepository walletRepository;

    public WalletQueryService(WalletRepository walletRepository) {
        this.walletRepository = walletRepository;
    }

    public WalletDTO getById(Long id) {
        return walletRepository.findById(id)
            .map(this::toDTO)
            .orElseThrow(() -> new IllegalArgumentException("Wallet not found"));
    }

    public WalletDTO getByUserIdAndChainType(String userId, String chainType) {
        return walletRepository.findByUserIdAndChainType(userId, chainType)
            .map(this::toDTO)
            .orElseThrow(() -> new IllegalArgumentException("Wallet not found"));
    }

    public List<WalletDTO> getByUserId(String userId) {
        return List.of();
    }

    private WalletDTO toDTO(Wallet wallet) {
        return new WalletDTO(wallet.getId(), wallet.getUserId(), wallet.getChainType(),
            wallet.getAddress(), wallet.getBalance(), wallet.getStatus());
    }
}