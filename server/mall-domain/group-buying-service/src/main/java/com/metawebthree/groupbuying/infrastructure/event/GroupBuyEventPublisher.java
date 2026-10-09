package com.metawebthree.groupbuying.infrastructure.event;

import org.springframework.stereotype.Component;

import com.metawebthree.groupbuying.domain.ports.GroupBuyEventPort;

import lombok.extern.slf4j.Slf4j;

/**
 * 拼团领域事件发布适配器（本地开发实现，仅记录日志）。
 */
@Slf4j
@Component
public class GroupBuyEventPublisher implements GroupBuyEventPort {

    @Override
    public void publishGroupBuyCreated(Long teamId, Long activityId, Long leaderId) {
        log.info("[event] GroupBuyCreated teamId={}, activityId={}, leaderId={}", teamId, activityId, leaderId);
    }

    @Override
    public void publishGroupBuyJoined(Long teamId, Long userId) {
        log.info("[event] GroupBuyJoined teamId={}, userId={}", teamId, userId);
    }

    @Override
    public void publishGroupBuySuccess(Long teamId, Long orderId) {
        log.info("[event] GroupBuySuccess teamId={}, orderId={}", teamId, orderId);
    }

    @Override
    public void publishGroupBuyFailed(Long teamId) {
        log.info("[event] GroupBuyFailed teamId={}", teamId);
    }

    @Override
    public void publishGroupBuyExpired(Long teamId) {
        log.info("[event] GroupBuyExpired teamId={}", teamId);
    }
}
