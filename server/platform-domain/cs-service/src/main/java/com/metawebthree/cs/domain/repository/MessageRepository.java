package com.metawebthree.cs.domain.repository;

import com.metawebthree.cs.domain.model.Message;

import java.util.List;

public interface MessageRepository {
    Message save(Message message);
    List<Message> findBySessionId(String sessionId);
    List<Message> findBySessionIdAfter(String sessionId, String afterMessageId);
}
