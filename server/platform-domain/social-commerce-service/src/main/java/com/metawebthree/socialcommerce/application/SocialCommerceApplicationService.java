package com.metawebthree.socialcommerce.application;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.socialcommerce.domain.model.ShareRewardConfigDO;
import com.metawebthree.socialcommerce.domain.model.ShareRecordDO;
import com.metawebthree.socialcommerce.domain.model.DistributionRelationDO;
import com.metawebthree.socialcommerce.domain.model.DistributionRewardDO;
import com.metawebthree.socialcommerce.domain.model.CommunityDO;
import com.metawebthree.socialcommerce.domain.model.CommunityMemberDO;
import com.metawebthree.socialcommerce.domain.repository.ShareRewardConfigRepository;
import com.metawebthree.socialcommerce.domain.repository.ShareRecordRepository;
import com.metawebthree.socialcommerce.domain.repository.DistributionRelationRepository;
import com.metawebthree.socialcommerce.domain.repository.DistributionRewardRepository;
import com.metawebthree.socialcommerce.domain.repository.CommunityRepository;
import com.metawebthree.socialcommerce.domain.repository.CommunityMemberRepository;
import com.metawebthree.socialcommerce.domain.ports.CommissionPort;
import com.metawebthree.socialcommerce.domain.ports.UserPort;
import com.metawebthree.socialcommerce.domain.ports.SocialCommerceEventPort;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
public class SocialCommerceApplicationService {
    private static final String STATUS_ACTIVE = "ACTIVE";
    private static final String STATUS_INACTIVE = "INACTIVE";
    private static final String STATUS_PENDING = "PENDING";
    private static final String STATUS_SETTLED = "SETTLED";
    private static final String ROLE_OWNER = "OWNER";
    private static final String ROLE_MEMBER = "MEMBER";
    private static final String ITEM_TYPE_PRODUCT = "PRODUCT";
    private static final String ITEM_TYPE_ORDER = "ORDER";

    private final ShareRewardConfigRepository configRepository;
    private final ShareRecordRepository shareRecordRepository;
    private final DistributionRelationRepository relationRepository;
    private final DistributionRewardRepository rewardRepository;
    private final CommunityRepository communityRepository;
    private final CommunityMemberRepository memberRepository;
    private final CommissionPort commissionPort;
    private final UserPort userPort;
    private final SocialCommerceEventPort eventPort;

    public SocialCommerceApplicationService(
            ShareRewardConfigRepository configRepository,
            ShareRecordRepository shareRecordRepository,
            DistributionRelationRepository relationRepository,
            DistributionRewardRepository rewardRepository,
            CommunityRepository communityRepository,
            CommunityMemberRepository memberRepository,
            CommissionPort commissionPort,
            UserPort userPort,
            SocialCommerceEventPort eventPort) {
        this.configRepository = configRepository;
        this.shareRecordRepository = shareRecordRepository;
        this.relationRepository = relationRepository;
        this.rewardRepository = rewardRepository;
        this.communityRepository = communityRepository;
        this.memberRepository = memberRepository;
        this.commissionPort = commissionPort;
        this.userPort = userPort;
        this.eventPort = eventPort;
    }

    @Transactional
    public Long createShareRewardConfig(String name, Integer rewardType,
            BigDecimal fixedAmount, BigDecimal percentage, Integer maxRewardCount,
            BigDecimal maxRewardAmount, LocalDateTime validFrom, LocalDateTime validTo) {
        Long configId = IdWorker.getId();
        ShareRewardConfigDO config = ShareRewardConfigDO.builder()
                .id(configId)
                .configName(name)
                .rewardType(rewardType)
                .fixedAmount(fixedAmount)
                .percentage(percentage)
                .maxRewardCount(maxRewardCount)
                .maxRewardAmount(maxRewardAmount)
                .status(1)
                .validFrom(java.sql.Timestamp.valueOf(validFrom))
                .validTo(java.sql.Timestamp.valueOf(validTo))
                .build();
        configRepository.save(config);
        log.info("Share reward config created: {}", configId);
        return configId;
    }

    @Transactional
    public String createShareRecord(Long sharerId, Long itemId, String itemType, String shareChannel) {
        if (!userPort.isValidUser(sharerId)) {
            throw new IllegalArgumentException("Invalid user");
        }
        Long recordId = IdWorker.getId();
        String shareUrl = "/share/" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
        ShareRecordDO record = ShareRecordDO.builder()
                .id(recordId)
                .sharerId(sharerId)
                .sharedItemId(itemId)
                .itemType(itemType)
                .shareChannel(shareChannel)
                .shareUrl(shareUrl)
                .clickCount(0)
                .purchaseCount(0)
                .rewardAmount(BigDecimal.ZERO)
                .status(STATUS_ACTIVE)
                .build();
        shareRecordRepository.save(record);
        eventPort.publishShareCreated(sharerId, itemId, itemType);
        log.info("Share record created: {} by user {}", recordId, sharerId);
        return shareUrl;
    }

    @Transactional
    public void recordShareClick(Long recordId) {
        ShareRecordDO record = shareRecordRepository.findById(recordId);
        if (record == null) {
            throw new IllegalArgumentException("Share record not found");
        }
        record.setClickCount(record.getClickCount() + 1);
        shareRecordRepository.update(record);
        eventPort.publishShareClicked(recordId);
        log.info("Share click recorded for: {}", recordId);
    }

    @Transactional
    public void settleShareReward(Long recordId, Long orderId, Long buyerId, BigDecimal orderAmount) {
        ShareRecordDO record = shareRecordRepository.findById(recordId);
        if (record == null) {
            throw new IllegalArgumentException("Share record not found");
        }
        ShareRewardConfigDO config = configRepository.findActiveConfig();
        if (config == null) {
            log.warn("No active reward config found");
            return;
        }
        BigDecimal reward = calculateReward(config, orderAmount, record.getClickCount());
        record.setPurchaseCount(record.getPurchaseCount() + 1);
        record.setRewardAmount(record.getRewardAmount().add(reward));
        shareRecordRepository.update(record);
        commissionPort.settleCommission(record.getSharerId(), reward, "SHARE_REWARD");
        eventPort.publishSharePurchased(recordId, orderId, buyerId);
        log.info("Share reward settled: {} for record {}", reward, recordId);
    }

    @Transactional
    public Boolean bindDistributionRelation(Long userId, Long referrerId) {
        if (userId.equals(referrerId)) {
            throw new IllegalArgumentException("Cannot bind to self");
        }
        if (!userPort.isValidUser(referrerId)) {
            throw new IllegalArgumentException("Invalid referrer");
        }
        if (relationRepository.existsByUserId(userId)) {
            throw new IllegalStateException("User already has a referrer");
        }
        DistributionRelationDO referrerRelation = relationRepository.findByUserId(referrerId);
        Integer level = referrerRelation != null ? referrerRelation.getLevel() + 1 : 1;
        if (level > 3) {
            throw new IllegalStateException("Maximum distribution level exceeded");
        }
        Long relationId = IdWorker.getId();
        Long rootReferrerId = referrerRelation != null ? referrerRelation.getRootReferrerId() : referrerId;
        DistributionRelationDO relation = DistributionRelationDO.builder()
                .id(relationId)
                .userId(userId)
                .referrerId(referrerId)
                .level(level)
                .rootReferrerId(rootReferrerId)
                .status(STATUS_ACTIVE)
                .bindTime(java.sql.Timestamp.valueOf(LocalDateTime.now()))
                .build();
        relationRepository.save(relation);
        eventPort.publishDistributionBind(userId, referrerId, level);
        log.info("Distribution relation bound: user {} -> referrer {} at level {}",
                userId, referrerId, level);
        return true;
    }

    @Transactional
    public void settleDistributionReward(Long orderId, Long buyerId, BigDecimal orderAmount) {
        DistributionRelationDO relation = relationRepository.findByUserId(buyerId);
        if (relation == null) {
            return;
        }
        DistributionRelationDO referrerRelation = relationRepository.findByUserId(relation.getReferrerId());
        BigDecimal commissionRate = referrerRelation != null 
                ? BigDecimal.valueOf(0.10).multiply(BigDecimal.valueOf(referrerRelation.getLevel()))
                : BigDecimal.valueOf(0.10);
        BigDecimal commission = orderAmount.multiply(commissionRate);
        DistributionRewardDO reward = DistributionRewardDO.builder()
                .id(IdWorker.getId())
                .referrerId(relation.getReferrerId())
                .buyerId(buyerId)
                .orderId(orderId)
                .orderAmount(orderAmount)
                .commissionAmount(commission)
                .level(relation.getLevel())
                .status(STATUS_SETTLED)
                .settledTime(java.sql.Timestamp.valueOf(LocalDateTime.now()))
                .build();
        rewardRepository.save(reward);
        commissionPort.settleCommission(relation.getReferrerId(), commission, "DISTRIBUTION");
        eventPort.publishCommissionSettled(relation.getReferrerId(), orderId, relation.getLevel());
        log.info("Distribution reward settled: {} for order {}", commission, orderId);
    }

    @Transactional
    public Long createCommunity(String name, String description, Long ownerId, String avatarUrl, Integer maxMembers) {
        if (!userPort.isValidUser(ownerId)) {
            throw new IllegalArgumentException("Invalid owner");
        }
        Long communityId = IdWorker.getId();
        String inviteCode = UUID.randomUUID().toString().replace("-", "").substring(0, 8).toUpperCase();
        CommunityDO community = CommunityDO.builder()
                .id(communityId)
                .communityName(name)
                .description(description)
                .ownerId(ownerId)
                .avatarUrl(avatarUrl)
                .memberCount(1)
                .maxMembers(maxMembers)
                .status(STATUS_ACTIVE)
                .inviteCode(inviteCode)
                .build();
        communityRepository.save(community);
        CommunityMemberDO ownerMember = CommunityMemberDO.builder()
                .id(IdWorker.getId())
                .communityId(communityId)
                .userId(ownerId)
                .role(ROLE_OWNER)
                .nickname(userPort.getUserNickname(ownerId))
                .messageCount(0)
                .status(STATUS_ACTIVE)
                .joinedAt(java.sql.Timestamp.valueOf(LocalDateTime.now()))
                .lastActiveAt(java.sql.Timestamp.valueOf(LocalDateTime.now()))
                .build();
        memberRepository.save(ownerMember);
        eventPort.publishCommunityCreated(communityId, ownerId);
        log.info("Community created: {} by user {}", communityId, ownerId);
        return communityId;
    }

    @Transactional
    public Boolean joinCommunity(Long communityId, Long userId) {
        CommunityDO community = communityRepository.findById(communityId);
        if (community == null) {
            throw new IllegalArgumentException("Community not found");
        }
        if (!STATUS_ACTIVE.equals(community.getStatus())) {
            throw new IllegalStateException("Community is not active");
        }
        if (community.getMemberCount() >= community.getMaxMembers()) {
            throw new IllegalStateException("Community is full");
        }
        if (memberRepository.existsByCommunityIdAndUserId(communityId, userId)) {
            throw new IllegalStateException("User already in community");
        }
        CommunityMemberDO member = CommunityMemberDO.builder()
                .id(IdWorker.getId())
                .communityId(communityId)
                .userId(userId)
                .role(ROLE_MEMBER)
                .nickname(userPort.getUserNickname(userId))
                .messageCount(0)
                .status(STATUS_ACTIVE)
                .joinedAt(java.sql.Timestamp.valueOf(LocalDateTime.now()))
                .lastActiveAt(java.sql.Timestamp.valueOf(LocalDateTime.now()))
                .build();
        memberRepository.save(member);
        community.setMemberCount(community.getMemberCount() + 1);
        communityRepository.update(community);
        eventPort.publishMemberJoined(communityId, userId);
        log.info("User {} joined community {}", userId, communityId);
        return true;
    }

    @Transactional
    public Boolean joinCommunityByCode(String inviteCode, Long userId) {
        CommunityDO community = communityRepository.findByInviteCode(inviteCode);
        if (community == null) {
            throw new IllegalArgumentException("Invalid invite code");
        }
        return joinCommunity(community.getId(), userId);
    }

    @Transactional
    public Boolean leaveCommunity(Long communityId, Long userId) {
        CommunityMemberDO member = memberRepository.findByCommunityIdAndUserId(communityId, userId);
        if (member == null) {
            throw new IllegalArgumentException("Member not found");
        }
        if (ROLE_OWNER.equals(member.getRole())) {
            throw new IllegalStateException("Owner cannot leave community");
        }
        memberRepository.delete(member.getId());
        CommunityDO community = communityRepository.findById(communityId);
        community.setMemberCount(community.getMemberCount() - 1);
        communityRepository.update(community);
        eventPort.publishMemberLeft(communityId, userId);
        log.info("User {} left community {}", userId, communityId);
        return true;
    }

    public List<ShareRecordDO> getUserShareRecords(Long userId) {
        return shareRecordRepository.findBySharerId(userId);
    }

    public List<DistributionRelationDO> getUserDistributionRelations(Long userId) {
        return relationRepository.findByReferrerId(userId);
    }

    public List<CommunityDO> getUserCommunities(Long userId) {
        List<CommunityMemberDO> members = memberRepository.findByUserId(userId);
        return members.stream()
                .map(m -> communityRepository.findById(m.getCommunityId()))
                .toList();
    }

    private BigDecimal calculateReward(ShareRewardConfigDO config, BigDecimal orderAmount, Integer clickCount) {
        BigDecimal reward;
        if (config.getFixedAmount() != null && config.getFixedAmount().compareTo(BigDecimal.ZERO) > 0) {
            reward = config.getFixedAmount();
        } else if (config.getPercentage() != null && config.getPercentage().compareTo(BigDecimal.ZERO) > 0) {
            reward = orderAmount.multiply(config.getPercentage());
        } else {
            reward = BigDecimal.ZERO;
        }
        if (config.getMaxRewardAmount() != null && reward.compareTo(config.getMaxRewardAmount()) > 0) {
            reward = config.getMaxRewardAmount();
        }
        return reward;
    }
}