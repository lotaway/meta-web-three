package com.metawebthree.groupbuying.domain.ports;

public interface GroupBuyEventPort {
    void publishGroupBuyCreated(Long teamId, Long activityId, Long leaderId);
    void publishGroupBuyJoined(Long teamId, Long userId);
    void publishGroupBuySuccess(Long teamId, Long orderId);
    void publishGroupBuyFailed(Long teamId);
    void publishGroupBuyExpired(Long teamId);
}