package com.metawebthree.reporting.domain.entity;

import java.time.LocalDateTime;

public class ReportSubscription {
    private Long id;
    private Long userId;
    private String userName;
    private ReportType reportType;
    private Frequency frequency;
    private String cronExpression;
    private Channel channel;
    private String recipient;
    private String webhookUrl;
    private Boolean enabled;
    private LocalDateTime nextSendTime;
    private LocalDateTime lastSendTime;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ReportType {
        SALES, INVENTORY, FINANCIAL
    }

    public enum Frequency {
        DAILY, WEEKLY, MONTHLY
    }

    public enum Channel {
        EMAIL, DINGTALK
    }

    public void calculateNextSendTime() {
        LocalDateTime now = LocalDateTime.now();
        switch (frequency) {
            case DAILY:
                this.nextSendTime = now.plusDays(1).withHour(8).withMinute(0).withSecond(0);
                break;
            case WEEKLY:
                this.nextSendTime = now.plusWeeks(1).withHour(8).withMinute(0).withSecond(0);
                break;
            case MONTHLY:
                this.nextSendTime = now.plusMonths(1).withDayOfMonth(1).withHour(8).withMinute(0).withSecond(0);
                break;
        }
    }

    public boolean shouldSend() {
        if (!enabled || nextSendTime == null) {
            return false;
        }
        return LocalDateTime.now().isAfter(nextSendTime) || LocalDateTime.now().equals(nextSendTime);
    }

    public void markAsSent() {
        this.lastSendTime = LocalDateTime.now();
        calculateNextSendTime();
    }

    public Long getId() { return id; }
    public Long getUserId() { return userId; }
    public String getUserName() { return userName; }
    public ReportType getReportType() { return reportType; }
    public Frequency getFrequency() { return frequency; }
    public String getCronExpression() { return cronExpression; }
    public Channel getChannel() { return channel; }
    public String getRecipient() { return recipient; }
    public String getWebhookUrl() { return webhookUrl; }
    public Boolean getEnabled() { return enabled; }
    public LocalDateTime getNextSendTime() { return nextSendTime; }
    public LocalDateTime getLastSendTime() { return lastSendTime; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }

    public void setId(Long id) { this.id = id; }
    public void setUserId(Long userId) { this.userId = userId; }
    public void setUserName(String userName) { this.userName = userName; }
    public void setReportType(ReportType reportType) { this.reportType = reportType; }
    public void setFrequency(Frequency frequency) { this.frequency = frequency; }
    public void setCronExpression(String cronExpression) { this.cronExpression = cronExpression; }
    public void setChannel(Channel channel) { this.channel = channel; }
    public void setRecipient(String recipient) { this.recipient = recipient; }
    public void setWebhookUrl(String webhookUrl) { this.webhookUrl = webhookUrl; }
    public void setEnabled(Boolean enabled) { this.enabled = enabled; }
    public void setNextSendTime(LocalDateTime nextSendTime) { this.nextSendTime = nextSendTime; }
    public void setLastSendTime(LocalDateTime lastSendTime) { this.lastSendTime = lastSendTime; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}