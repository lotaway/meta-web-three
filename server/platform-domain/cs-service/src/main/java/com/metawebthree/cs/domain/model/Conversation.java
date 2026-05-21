package com.metawebthree.cs.domain.model;

import com.metawebthree.cs.domain.model.enums.ConversationStatus;
import com.metawebthree.cs.domain.model.enums.ChannelType;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Data
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
}
