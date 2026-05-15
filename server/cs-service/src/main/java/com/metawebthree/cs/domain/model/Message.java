package com.metawebthree.cs.domain.model;

import com.metawebthree.cs.domain.model.enums.MessageType;
import com.metawebthree.cs.domain.model.enums.SenderType;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Data
public class Message {
    private String id;
    private String sessionId;
    private String messageId;
    private SenderType senderType;
    private Long senderId;
    private MessageType msgType;
    private String content;
    private Map<String, Object> extra;
    private LocalDateTime timestamp;
    private Boolean readStatus;

    public Message() {}

    public Message(String sessionId, String messageId, SenderType senderType,
                   Long senderId, MessageType msgType, String content) {
        this.sessionId = sessionId;
        this.messageId = messageId;
        this.senderType = senderType;
        this.senderId = senderId;
        this.msgType = msgType;
        this.content = content;
        this.timestamp = LocalDateTime.now();
        this.readStatus = false;
    }
}
