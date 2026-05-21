package com.metawebthree.cs.domain.ports;

import com.metawebthree.cs.domain.model.enums.ConversationEvent;

public interface ConversationEventPort {
    void publish(String sessionId, ConversationEvent event, String detail);
}
