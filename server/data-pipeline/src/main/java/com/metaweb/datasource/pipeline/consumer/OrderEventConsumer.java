package com.metaweb.datasource.pipeline.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metaweb.datasource.pipeline.model.OrderEvent;
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
public class OrderEventConsumer {

    @Autowired
    private EtlService etlService;

    @Autowired
    private ObjectMapper objectMapper;

    @KafkaListener(
        topics = "${etl.topics.order}",
        containerFactory = "kafkaListenerContainerFactory"
    )
    public void consumeOrderEvents(List<ConsumerRecord<String, String>> records, Acknowledgment ack) {
        log.info("Received {} order events", records.size());
        try {
            for (ConsumerRecord<String, String> record : records) {
                processSingleEvent(record);
            }
            ack.acknowledge();
            log.info("Successfully processed {} order events", records.size());
        } catch (Exception e) {
            log.error("Batch processing failed for order events", e);
            throw e;
        }
    }

    private void processSingleEvent(ConsumerRecord<String, String> record) {
        try {
            OrderEvent event = objectMapper.readValue(record.value(), OrderEvent.class);
            etlService.processOrderEvent(event);
        } catch (Exception e) {
            log.error("Failed to process order event: {}", record.value(), e);
        }
    }
}
