package com.metawebthree.socialcommerce.domain.ports;

public interface SocialCommerceEventPort {
    void publishShareCreated(Long sharerId, Long itemId, String itemType);
    void publishShareClicked(Long recordId);
    void publishSharePurchased(Long recordId, Long orderId, Long buyerId);
    void publishDistributionBind(Long userId, Long referrerId, Integer level);
    void publishCommissionSettled(Long referrerId, Long orderId, Integer level);
    void publishCommunityCreated(Long communityId, Long ownerId);
    void publishMemberJoined(Long communityId, Long userId);
    void publishMemberLeft(Long communityId, Long userId);
}