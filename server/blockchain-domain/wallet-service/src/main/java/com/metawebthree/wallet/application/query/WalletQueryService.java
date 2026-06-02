package com.metawebthree.wallet.application.query;

import com.metawebthree.wallet.application.dto.WalletDTO;
import com.metawebthree.wallet.domain.entity.Wallet;
import com.metawebthree.wallet.domain.repository.WalletRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
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

    public Map<String, Object> listWallets(Integer pageNum, Integer pageSize, String userId, String chainType, String status) {
        List<Wallet> allWallets = walletRepository.findAll();

        // Apply filters
        if (userId != null && !userId.isEmpty()) {
            allWallets = allWallets.stream()
                    .filter(w -> w.getUserId() != null && w.getUserId().toLowerCase().contains(userId.toLowerCase()))
                    .collect(Collectors.toList());
        }
        if (chainType != null && !chainType.isEmpty()) {
            allWallets = allWallets.stream()
                    .filter(w -> w.getChainType() != null && w.getChainType().equalsIgnoreCase(chainType))
                    .collect(Collectors.toList());
        }
        if (status != null && !status.isEmpty()) {
            allWallets = allWallets.stream()
                    .filter(w -> w.getStatus() != null && w.getStatus().equalsIgnoreCase(status))
                    .collect(Collectors.toList());
        }

        // Sort by createdAt desc
        allWallets.sort((a, b) -> {
            if (b.getCreatedAt() == null) return -1;
            if (a.getCreatedAt() == null) return 1;
            return b.getCreatedAt().compareTo(a.getCreatedAt());
        });

        int total = allWallets.size();
        int start = (pageNum - 1) * pageSize;
        int end = Math.min(start + pageSize, total);
        List<WalletDTO> pageData = allWallets.subList(Math.min(start, total), Math.min(end, total))
                .stream()
                .map(this::toDTO)
                .collect(Collectors.toList());

        return Map.of(
            "list", pageData,
            "total", total,
            "pageNum", pageNum,
            "pageSize", pageSize
        );
    }

    public Map<String, Object> getStatistics() {
        List<Wallet> allWallets = walletRepository.findAll();
        
        long totalCount = allWallets.size();
        long activeCount = allWallets.stream().filter(w -> "ACTIVE".equalsIgnoreCase(w.getStatus())).count();
        long frozenCount = allWallets.stream().filter(w -> "FROZEN".equalsIgnoreCase(w.getStatus())).count();
        BigDecimal totalBalance = allWallets.stream()
                .map(Wallet::getBalance)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        // Count by chain type
        Map<String, Long> chainTypeCount = allWallets.stream()
                .collect(Collectors.groupingBy(w -> w.getChainType() != null ? w.getChainType() : "UNKNOWN", Collectors.counting()));
        
        return Map.of(
            "totalCount", totalCount,
            "activeCount", activeCount,
            "frozenCount", frozenCount,
            "totalBalance", totalBalance,
            "chainTypeCount", chainTypeCount
        );
    }

    private WalletDTO toDTO(Wallet wallet) {
        return new WalletDTO(wallet.getId(), wallet.getUserId(), wallet.getChainType(),
            wallet.getAddress(), wallet.getBalance(), wallet.getStatus());
    }
}