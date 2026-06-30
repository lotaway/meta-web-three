package com.metawebthree.digitaltwin.kafka;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.DltHandler;
import org.springframework.kafka.support.KafkaHeaders;
import org.springframework.messaging.handler.annotation.Header;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Component;

/**
 * Dead Letter Topic (DLT) consumer for handling failed Kafka messages.
 * This handler receives messages that have exhausted all retry attempts.
 */
@Component
public class DigitalTwinDltConsumer {

    private static final Logger logger = LoggerFactory.getLogger(DigitalTwinDltConsumer.class);
    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Handler for dead letter messages.
     * Logs failed messages for manual inspection and potential reprocessing.
     */
    @DltHandler
    public void handleDlt(
            @Payload String message,
            @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
            @Header(KafkaHeaders.EXCEPTION_MESSAGE) String exceptionMessage,
            @Header(KafkaHeaders.EXCEPTION_STACKTRACE) String stackTrace) {
        
        logger.error("=== DLT Message Received ===");
        logger.error("Original Topic: {}", topic);
        logger.error("Exception Message: {}", exceptionMessage);
        logger.error("Message Payload: {}", message);
        logger.error("Stack Trace: {}", stackTrace);
        logger.error("=============================");
    }
}