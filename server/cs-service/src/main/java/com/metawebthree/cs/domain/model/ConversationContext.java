package com.metawebthree.cs.domain.model;

import lombok.Data;

@Data
public class ConversationContext {
    private String sessionId;
    private Long productId;
    private Long orderId;
    private String productName;
    private String orderStatus;

    public ConversationContext(String sessionId) {
        this.sessionId = sessionId;
    }
}
