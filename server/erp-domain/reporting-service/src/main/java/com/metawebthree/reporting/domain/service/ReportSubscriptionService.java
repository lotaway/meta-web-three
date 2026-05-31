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

@Service
@RequiredArgsConstructor
public class ReportSubscriptionService {

    private final ReportSubscriptionRepository subscriptionRepository;

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

    public void toggleSubscription(Long id, boolean enabled) {
        ReportSubscription subscription = subscriptionRepository.findById(id);
        if (subscription != null) {
            subscription.setEnabled(enabled);
            subscriptionRepository.update(subscription);
        }
    }

    public void deleteSubscription(Long id) {
        subscriptionRepository.delete(id);
    }

    public List<ReportSubscription> getUserSubscriptions(Long userId) {
        return subscriptionRepository.findByUserId(userId);
    }

    public List<ReportSubscription> getEnabledSubscriptions() {
        return subscriptionRepository.findEnabled();
    }

    public List<ReportSubscription> getDueSubscriptions() {
        return subscriptionRepository.findDueSubscriptions(LocalDateTime.now());
    }

    public void markAsSent(Long id) {
        ReportSubscription subscription = subscriptionRepository.findById(id);
        if (subscription != null) {
            subscription.markAsSent();
            subscriptionRepository.update(subscription);
        }
    }

    public ReportSubscription getSubscription(Long id) {
        return subscriptionRepository.findById(id);
    }
}