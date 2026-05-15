package com.metawebthree.cs.domain.model;

import com.metawebthree.cs.domain.model.enums.MessageType;
import com.metawebthree.cs.domain.model.enums.SenderType;

import java.time.LocalDateTime;
import java.util.Map;

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

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    public String getMessageId() { return messageId; }
    public void setMessageId(String messageId) { this.messageId = messageId; }
    public SenderType getSenderType() { return senderType; }
    public void setSenderType(SenderType senderType) { this.senderType = senderType; }
    public Long getSenderId() { return senderId; }
    public void setSenderId(Long senderId) { this.senderId = senderId; }
    public MessageType getMsgType() { return msgType; }
    public void setMsgType(MessageType msgType) { this.msgType = msgType; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    public Map<String, Object> getExtra() { return extra; }
    public void setExtra(Map<String, Object> extra) { this.extra = extra; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
    public Boolean getReadStatus() { return readStatus; }
    public void setReadStatus(Boolean readStatus) { this.readStatus = readStatus; }
}
