package com.metawebthree.wallet.application.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.wallet.application.dto.ActivityDTO;
import com.metawebthree.wallet.application.dto.CreateActivityRequest;
import com.metawebthree.wallet.domain.entity.SolanaActivity;
import com.metawebthree.wallet.infrastructure.persistence.repository.SolanaActivityMapper;
import com.metawebthree.wallet.infrastructure.solana.SolanaContractClient;
import com.metawebthree.wallet.infrastructure.solana.SolanaWalletManager;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class SolanaActivityService {

    private final SolanaContractClient contractClient;
    private final SolanaActivityMapper activityMapper;
    private final SolanaWalletManager walletManager;

    public SolanaActivityService(SolanaContractClient contractClient, SolanaActivityMapper activityMapper, SolanaWalletManager walletManager) {
        this.contractClient = contractClient;
        this.activityMapper = activityMapper;
        this.walletManager = walletManager;
    }

    public ActivityDTO createActivity(CreateActivityRequest request) {
        if (request.getStartTime() == null || request.getEndTime() == null) {
            throw new IllegalArgumentException("Start and end time are required");
        }
        if (request.getStartTime() >= request.getEndTime()) {
            throw new IllegalArgumentException("Start time must be before end time");
        }
        if (request.getEntryFee() == null || request.getEntryFee() <= 0) {
            throw new IllegalArgumentException("Entry fee must be positive");
        }

        byte[] authority = SolanaContractClient.decodeBase58(request.getAuthority());
        byte[] privateKey = walletManager.getPrivateKey(request.getAuthority());

        Integer[] pcts = request.getRewardPercentages() != null
            ? request.getRewardPercentages()
            : new Integer[]{5000, 3000, 2000};
        int[] rewardPcts = new int[]{pcts[0], pcts[1], pcts[2]};

        String txSig = contractClient.createActivity(authority,
            request.getStartTime(), request.getEndTime(),
            request.getEntryFee(), rewardPcts, privateKey);

        byte[] activity = contractClient.deriveActivityAddress(authority);
        String activityAddress = SolanaContractClient.base58Encode(activity);

        SolanaActivity entity = new SolanaActivity();
        entity.setActivityAddress(activityAddress);
        entity.setAuthorityAddress(request.getAuthority());
        entity.setStartTime(request.getStartTime());
        entity.setEndTime(request.getEndTime());
        entity.setEntryFee(request.getEntryFee());
        entity.setRewardPcts(Arrays.toString(pcts));
        entity.setPaymentMint("So11111111111111111111111111111111111111112");
        entity.setTotalPool(0L);
        entity.setParticipantCount(0);
        entity.setTxSignature(txSig);
        entity.setCreatedAt(LocalDateTime.now());
        entity.setUpdatedAt(LocalDateTime.now());
        activityMapper.insert(entity);

        return toDTO(entity);
    }

    public List<ActivityDTO> listActivities() {
        return activityMapper.selectList(null).stream()
            .map(this::toDTO)
            .collect(Collectors.toList());
    }

    public ActivityDTO getActivity(String activityAddress) {
        SolanaActivity entity = activityMapper.selectOne(
            new LambdaQueryWrapper<SolanaActivity>()
                .eq(SolanaActivity::getActivityAddress, activityAddress));
        return entity != null ? toDTO(entity) : null;
    }

    public ActivityDTO participate(String activityAddress, String participant, String authorityAddress) {
        byte[] participantKey = SolanaContractClient.decodeBase58(participant);
        byte[] authority = SolanaContractClient.decodeBase58(authorityAddress);
        byte[] paymentMint = SolanaContractClient.decodeBase58("So11111111111111111111111111111111111111112");
        byte[] privateKey = walletManager.getPrivateKey(participant);

        String txSig = contractClient.participateActivity(participantKey, authority, paymentMint, privateKey);

        SolanaActivity entity = activityMapper.selectOne(
            new LambdaQueryWrapper<SolanaActivity>()
                .eq(SolanaActivity::getActivityAddress, activityAddress));
        if (entity != null) {
            entity.setParticipantCount(entity.getParticipantCount() + 1);
            entity.setUpdatedAt(LocalDateTime.now());
            activityMapper.updateById(entity);
        }

        return new ActivityDTO(activityAddress, "", 0L, 0L, 0L, new Integer[]{0, 0, 0}, 0L, 0L, txSig);
    }

    public String claimReward(String activityAddress, String winner, Integer rank, String authorityAddress) {
        byte[] winnerKey = SolanaContractClient.decodeBase58(winner);
        byte[] authority = SolanaContractClient.decodeBase58(authorityAddress);
        byte[] paymentMint = SolanaContractClient.decodeBase58("So11111111111111111111111111111111111111112");
        byte[] privateKey = walletManager.getPrivateKey(authorityAddress);

        String txSig = contractClient.claimReward(winnerKey, authority, paymentMint,
            rank != null ? rank : 1, List.of(), privateKey);

        return txSig;
    }

    private ActivityDTO toDTO(SolanaActivity entity) {
        Integer[] rewardPcts = parseRewardPcts(entity.getRewardPcts());
        return new ActivityDTO(
            entity.getActivityAddress(),
            entity.getAuthorityAddress(),
            entity.getStartTime(),
            entity.getEndTime(),
            entity.getEntryFee(),
            rewardPcts,
            entity.getTotalPool(),
            entity.getParticipantCount() != null ? entity.getParticipantCount().longValue() : 0L,
            entity.getTxSignature()
        );
    }

    private Integer[] parseRewardPcts(String json) {
        if (json == null || json.isBlank()) {
            return new Integer[]{5000, 3000, 2000};
        }
        try {
            String trimmed = json.replaceAll("[\\[\\] ]", "");
            String[] parts = trimmed.split(",");
            return Arrays.stream(parts).map(Integer::parseInt).toArray(Integer[]::new);
        } catch (Exception e) {
            return new Integer[]{5000, 3000, 2000};
        }
    }
}
