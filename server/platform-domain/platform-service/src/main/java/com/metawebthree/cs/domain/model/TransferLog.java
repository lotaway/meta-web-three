package com.metawebthree.cs.domain.model;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class TransferLog {
    private Long id;
    private String sessionId;
    private Long fromAgentId;
    private Long toAgentId;
    private String reason;
    private LocalDateTime transferTime;

    public TransferLog() {}

    public TransferLog(String sessionId, Long fromAgentId, Long toAgentId, String reason) {
        this.sessionId = sessionId;
        this.fromAgentId = fromAgentId;
        this.toAgentId = toAgentId;
        this.reason = reason;
        this.transferTime = LocalDateTime.now();
    }
}