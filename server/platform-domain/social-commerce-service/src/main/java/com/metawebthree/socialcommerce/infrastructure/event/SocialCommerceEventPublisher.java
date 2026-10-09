package com.metawebthree.socialcommerce.infrastructure.event;

import org.springframework.stereotype.Component;

import com.metawebthree.socialcommerce.domain.ports.SocialCommerceEventPort;

import lombok.extern.slf4j.Slf4j;

/**
 * 社交电商领域事件发布适配器（本地开发实现，仅记录日志）。
 */
@Slf4j
@Component
public class SocialCommerceEventPublisher implements SocialCommerceEventPort {

    @Override
    public void publishShareCreated(Long sharerId, Long itemId, String itemType) {
        log.info("[event] ShareCreated sharerId={}, itemId={}, itemType={}", sharerId, itemId, itemType);
    }

    @Override
    public void publishShareClicked(Long recordId) {
        log.info("[event] ShareClicked recordId={}", recordId);
    }

    @Override
    public void publishSharePurchased(Long recordId, Long orderId, Long buyerId) {
        log.info("[event] SharePurchased recordId={}, orderId={}, buyerId={}", recordId, orderId, buyerId);
    }

    @Override
    public void publishDistributionBind(Long userId, Long referrerId, Integer level) {
        log.info("[event] DistributionBind userId={}, referrerId={}, level={}", userId, referrerId, level);
    }

    @Override
    public void publishCommissionSettled(Long referrerId, Long orderId, Integer level) {
        log.info("[event] CommissionSettled referrerId={}, orderId={}, level={}", referrerId, orderId, level);
    }

    @Override
    public void publishCommunityCreated(Long communityId, Long ownerId) {
        log.info("[event] CommunityCreated communityId={}, ownerId={}", communityId, ownerId);
    }

    @Override
    public void publishMemberJoined(Long communityId, Long userId) {
        log.info("[event] MemberJoined communityId={}, userId={}", communityId, userId);
    }

    @Override
    public void publishMemberLeft(Long communityId, Long userId) {
        log.info("[event] MemberLeft communityId={}, userId={}", communityId, userId);
    }
}
