package com.metawebthree.cs.infrastructure.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.common.utils.RocketMQ.MQProducer;
import com.metawebthree.cs.domain.model.enums.ConversationEvent;
import com.metawebthree.cs.domain.ports.ConversationEventPort;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.Map;

public class RocketMQConversationEventPort implements ConversationEventPort {
    private static final Logger log = LoggerFactory.getLogger(RocketMQConversationEventPort.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final String TOPIC = "CS_CONVERSATION_EVENT";

    private final MQProducer mqProducer;

    public RocketMQConversationEventPort(MQProducer mqProducer) {
        this.mqProducer = mqProducer;
    }

    @Override
    public void publish(String sessionId, ConversationEvent event, String detail) {
        try {
            String payload = objectMapper.writeValueAsString(Map.of(
                    "sessionId", sessionId,
                    "event", event.name(),
                    "detail", detail,
                    "timestamp", Instant.now().toString()));
            mqProducer.send(TOPIC, payload, event.name());
        } catch (Exception e) {
            log.error("failed to publish event session:{} event:{}", sessionId, event, e);
        }
    }
}
