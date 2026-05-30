package com.metawebthree.reporting.domain.service;

import com.metawebthree.reporting.domain.entity.ReportSubscription;
import com.metawebthree.reporting.domain.entity.ReportSubscription.Channel;
import com.metawebthree.reporting.domain.entity.ReportSubscription.Frequency;
import com.metawebthree.reporting.domain.entity.ReportSubscription.ReportType;
import com.metawebthree.reporting.domain.repository.ReportSubscriptionRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 报表订阅领域服务
 */
@Service
@RequiredArgsConstructor
public class ReportSubscriptionService {

    private final ReportSubscriptionRepository subscriptionRepository;

    /**
     * 创建订阅
     */
    public Long createSubscription(Long userId, String userName, ReportType reportType,
                                    Frequency frequency, Channel channel, String recipient) {
        ReportSubscription subscription = new ReportSubscription();
        subscription.setUserId(userId);
        subscription.setUserName(userName);
        subscription.setReportType(reportType);
        subscription.setFrequency(frequency);
        subscription.setChannel(channel);
        subscription.setRecipient(recipient);
        subscription.setEnabled(true);
        subscription.calculateNextSendTime();
        subscription.setCreatedAt(LocalDateTime.now());
        
        return subscriptionRepository.save(subscription);
    }

    /**
     * 创建钉钉 webhook 订阅
     */
    public Long createDingTalkSubscription(Long userId, String userName, ReportType reportType,
                                            Frequency frequency, String webhookUrl) {
        ReportSubscription subscription = new ReportSubscription();
        subscription.setUserId(userId);
        subscription.setUserName(userName);
        subscription.setReportType(reportType);
        subscription.setFrequency(frequency);
        subscription.setChannel(Channel.DINGTALK);
        subscription.setWebhookUrl(webhookUrl);
        subscription.setEnabled(true);
        subscription.calculateNextSendTime();
        subscription.setCreatedAt(LocalDateTime.now());
        
        return subscriptionRepository.save(subscription);
    }

    /**
     * 更新订阅
     */
    public void updateSubscription(Long id, Frequency frequency, Channel channel, 
                                    String recipient, Boolean enabled) {
        ReportSubscription subscription = subscriptionRepository.findById(id);
        if (subscription != null) {
            if (frequency != null) {
                subscription.setFrequency(frequency);
            }
            if (channel != null) {
                subscription.setChannel(channel);
            }
            if (recipient != null) {
                subscription.setRecipient(recipient);
            }
            if (enabled != null) {
                subscription.setEnabled(enabled);
            }
            subscriptionRepository.update(subscription);
        }
    }

    /**
     * 启用/禁用订阅
     */
    public void toggleSubscription(Long id, boolean enabled) {
        ReportSubscription subscription = subscriptionRepository.findById(id);
        if (subscription != null) {
            subscription.setEnabled(enabled);
            subscriptionRepository.update(subscription);
        }
    }

    /**
     * 删除订阅
     */
    public void deleteSubscription(Long id) {
        subscriptionRepository.delete(id);
    }

    /**
     * 获取用户的订阅列表
     */
    public List<ReportSubscription> getUserSubscriptions(Long userId) {
        return subscriptionRepository.findByUserId(userId);
    }

    /**
     * 获取所有启用的订阅
     */
    public List<ReportSubscription> getEnabledSubscriptions() {
        return subscriptionRepository.findEnabled();
    }

    /**
     * 获取需要发送的订阅
     */
    public List<ReportSubscription> getDueSubscriptions() {
        return subscriptionRepository.findDueSubscriptions(LocalDateTime.now());
    }

    /**
     * 标记订阅已发送
     */
    public void markAsSent(Long id) {
        ReportSubscription subscription = subscriptionRepository.findById(id);
        if (subscription != null) {
            subscription.markAsSent();
            subscriptionRepository.update(subscription);
        }
    }

    /**
     * 获取订阅详情
     */
    public ReportSubscription getSubscription(Long id) {
        return subscriptionRepository.findById(id);
    }
}