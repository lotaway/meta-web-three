package com.metawebthree.wallet.application.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.wallet.application.dto.CommissionDTO;
import com.metawebthree.wallet.domain.entity.SolanaCommissionRelation;
import com.metawebthree.wallet.infrastructure.persistence.repository.SolanaCommissionRelationMapper;
import com.metawebthree.wallet.infrastructure.solana.SolanaContractClient;
import com.metawebthree.wallet.infrastructure.solana.SolanaWalletManager;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
public class SolanaCommissionService {

    private final SolanaContractClient contractClient;
    private final SolanaCommissionRelationMapper commissionMapper;
    private final SolanaWalletManager walletManager;

    public SolanaCommissionService(SolanaContractClient contractClient, SolanaCommissionRelationMapper commissionMapper, SolanaWalletManager walletManager) {
        this.contractClient = contractClient;
        this.commissionMapper = commissionMapper;
        this.walletManager = walletManager;
    }

    public CommissionDTO setUpline(String target, String upline) {
        if (target.equals(upline)) {
            throw new IllegalArgumentException("Cannot set self as upline");
        }

        byte[] targetKey = SolanaContractClient.decodeBase58(target);
        byte[] uplineKey = SolanaContractClient.decodeBase58(upline);
        byte[] privateKey = walletManager.getPrivateKey(target);

        String txSig = contractClient.setUpline(targetKey, uplineKey, targetKey, privateKey);

        byte[] graph = contractClient.deriveCommissionGraph(targetKey);
        String graphAddress = SolanaContractClient.base58Encode(graph);

        SolanaCommissionRelation existing = commissionMapper.selectOne(
            new LambdaQueryWrapper<SolanaCommissionRelation>()
                .eq(SolanaCommissionRelation::getAccountAddress, target));
        if (existing != null) {
            existing.setUplineAddress(upline);
            existing.setTxSignature(txSig);
            existing.setUpdatedAt(LocalDateTime.now());
            commissionMapper.updateById(existing);

            long downlineCount = commissionMapper.selectCount(
                new LambdaQueryWrapper<SolanaCommissionRelation>()
                    .eq(SolanaCommissionRelation::getUplineAddress, upline));
            SolanaCommissionRelation uplineEntity = commissionMapper.selectOne(
                new LambdaQueryWrapper<SolanaCommissionRelation>()
                    .eq(SolanaCommissionRelation::getAccountAddress, upline));
            if (uplineEntity != null) {
                uplineEntity.setDownlineCount((int) downlineCount);
                uplineEntity.setUpdatedAt(LocalDateTime.now());
                commissionMapper.updateById(uplineEntity);
            }

            return new CommissionDTO(graphAddress, upline, existing.getLevel(), 0);
        }

        SolanaCommissionRelation entity = new SolanaCommissionRelation();
        entity.setAccountAddress(target);
        entity.setUplineAddress(upline);
        entity.setLevel(1);
        entity.setDownlineCount(0);
        entity.setTxSignature(txSig);
        entity.setCreatedAt(LocalDateTime.now());
        entity.setUpdatedAt(LocalDateTime.now());
        commissionMapper.insert(entity);

        long downlineCount = commissionMapper.selectCount(
            new LambdaQueryWrapper<SolanaCommissionRelation>()
                .eq(SolanaCommissionRelation::getUplineAddress, upline));
        SolanaCommissionRelation uplineEntity = commissionMapper.selectOne(
            new LambdaQueryWrapper<SolanaCommissionRelation>()
                .eq(SolanaCommissionRelation::getAccountAddress, upline));
        if (uplineEntity != null) {
                uplineEntity.setDownlineCount((int) downlineCount);
                uplineEntity.setUpdatedAt(LocalDateTime.now());
            commissionMapper.updateById(uplineEntity);
        }

        return new CommissionDTO(graphAddress, upline, 1, 0);
    }

    public CommissionDTO getCommissionGraph(String account) {
        SolanaCommissionRelation entity = commissionMapper.selectOne(
            new LambdaQueryWrapper<SolanaCommissionRelation>()
                .eq(SolanaCommissionRelation::getAccountAddress, account));
        if (entity != null) {
            byte[] graph = contractClient.deriveCommissionGraph(SolanaContractClient.decodeBase58(account));
            return new CommissionDTO(
                SolanaContractClient.base58Encode(graph),
                entity.getUplineAddress(),
                entity.getLevel(),
                entity.getDownlineCount()
            );
        }
        return new CommissionDTO(account, "", 0, 0);
    }

    public String distributeCommission(String sellerAddress, String paymentMintAddress, Long saleAmount) {
        if (saleAmount == null || saleAmount <= 0) {
            throw new IllegalArgumentException("Sale amount must be positive");
        }

        byte[] seller = SolanaContractClient.decodeBase58(sellerAddress);
        byte[] paymentMint = SolanaContractClient.decodeBase58(paymentMintAddress);
        byte[] privateKey = walletManager.getPrivateKey(sellerAddress);

        SolanaCommissionRelation relation = commissionMapper.selectOne(
            new LambdaQueryWrapper<SolanaCommissionRelation>()
                .eq(SolanaCommissionRelation::getAccountAddress, sellerAddress));
        if (relation == null) {
            throw new IllegalArgumentException("No upline set for seller: " + sellerAddress);
        }

        byte[] uplineAta = contractClient.findAssociatedTokenAddress(
            paymentMint, SolanaContractClient.decodeBase58(relation.getUplineAddress()));

        return contractClient.distributeCommission(seller, paymentMint, saleAmount, privateKey, uplineAta);
    }
}
