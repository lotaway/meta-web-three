package com.metawebthree.cs.domain.model;

import com.metawebthree.cs.domain.model.enums.ConversationStatus;
import com.metawebthree.cs.domain.model.enums.ChannelType;

import java.time.LocalDateTime;
import java.util.Map;

public class Conversation {
    private String id;
    private String sessionId;
    private Long customerId;
    private Long agentId;
    private ConversationStatus status;
    private ChannelType channel;
    private Long productId;
    private Long orderId;
    private Integer queuePosition;
    private LocalDateTime createTime;
    private LocalDateTime activeTime;
    private LocalDateTime endTime;
    private Integer satisfactionScore;
    private Map<String, Object> metadata;

    public Conversation() {}

    public Conversation(String sessionId, Long customerId, ChannelType channel) {
        this.sessionId = sessionId;
        this.customerId = customerId;
        this.channel = channel;
        this.status = ConversationStatus.QUEUING;
        this.createTime = LocalDateTime.now();
        this.activeTime = LocalDateTime.now();
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    public Long getCustomerId() { return customerId; }
    public void setCustomerId(Long customerId) { this.customerId = customerId; }
    public Long getAgentId() { return agentId; }
    public void setAgentId(Long agentId) { this.agentId = agentId; }
    public ConversationStatus getStatus() { return status; }
    public void setStatus(ConversationStatus status) { this.status = status; }
    public ChannelType getChannel() { return channel; }
    public void setChannel(ChannelType channel) { this.channel = channel; }
    public Long getProductId() { return productId; }
    public void setProductId(Long productId) { this.productId = productId; }
    public Long getOrderId() { return orderId; }
    public void setOrderId(Long orderId) { this.orderId = orderId; }
    public Integer getQueuePosition() { return queuePosition; }
    public void setQueuePosition(Integer queuePosition) { this.queuePosition = queuePosition; }
    public LocalDateTime getCreateTime() { return createTime; }
    public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
    public LocalDateTime getActiveTime() { return activeTime; }
    public void setActiveTime(LocalDateTime activeTime) { this.activeTime = activeTime; }
    public LocalDateTime getEndTime() { return endTime; }
    public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
    public Integer getSatisfactionScore() { return satisfactionScore; }
    public void setSatisfactionScore(Integer satisfactionScore) { this.satisfactionScore = satisfactionScore; }
    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
}
