package com.metaweb.datasource.pipeline.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metaweb.datasource.pipeline.model.UserBehaviorEvent;
import com.metaweb.datasource.pipeline.service.EtlService;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.stereotype.Component;

import java.util.List;

@Slf4j
@Component
public class UserBehaviorEventConsumer {
    
    @Autowired
    private EtlService etlService;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @KafkaListener(
        topics = "${etl.topics.user-behavior}",
        containerFactory = "kafkaListenerContainerFactory"
    )
    public void consumeUserBehaviorEvents(List<ConsumerRecord<String, String>> records, Acknowledgment ack) {
        log.info("Received {} user behavior events", records.size());
        
        try {
            for (ConsumerRecord<String, String> record : records) {
                try {
                    String jsonValue = record.value();
                    UserBehaviorEvent event = objectMapper.readValue(jsonValue, UserBehaviorEvent.class);
                    
                    log.debug("Processing user behavior event: {}", event.getEventId());
                    
                    // Process and transform the event
                    etlService.processUserBehaviorEvent(event);
                    
                } catch (Exception e) {
                    log.error("Failed to process user behavior event: {}", record.value(), e);
                    // In production, send to dead letter queue
                }
            }
            
            // Manually acknowledge after successful processing
            ack.acknowledge();
            log.info("Successfully processed {} user behavior events", records.size());
            
        } catch (Exception e) {
            log.error("Batch processing failed for user behavior events", e);
            throw e;
        }
    }
}
