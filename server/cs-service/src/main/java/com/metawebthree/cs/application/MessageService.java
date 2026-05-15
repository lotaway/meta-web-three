package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.Message;
import com.metawebthree.cs.domain.model.enums.MessageType;
import com.metawebthree.cs.domain.model.enums.SenderType;
import com.metawebthree.cs.domain.repository.MessageRepository;

import java.util.List;
import java.util.Map;
import java.util.UUID;

public class MessageService {
    private final MessageRepository messageRepository;

    public MessageService(MessageRepository messageRepository) {
        this.messageRepository = messageRepository;
    }

    public Message send(String sessionId, SenderType senderType, Long senderId,
                         MessageType msgType, String content) {
        Message message = new Message(sessionId, UUID.randomUUID().toString(),
                senderType, senderId, msgType, content);
        return messageRepository.save(message);
    }

    public Message sendWithExtra(String sessionId, SenderType senderType, Long senderId,
                                  MessageType msgType, String content, Map<String, Object> extra) {
        Message message = new Message(sessionId, UUID.randomUUID().toString(),
                senderType, senderId, msgType, content);
        message.setExtra(extra);
        return messageRepository.save(message);
    }

    public List<Message> listBySession(String sessionId) {
        return messageRepository.findBySessionId(sessionId);
    }

    public List<Message> listAfter(String sessionId, String afterMessageId) {
        return messageRepository.findBySessionIdAfter(sessionId, afterMessageId);
    }
}
