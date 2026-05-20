package com.metawebthree.event;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.kafka.core.KafkaTemplate;

/**
 * Auto-configuration for Event SDK.
 */
@AutoConfiguration
public class EventAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public ObjectMapper eventObjectMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(new JavaTimeModule());
        return mapper;
    }

    @Bean
    @ConditionalOnMissingBean
    public EventPublisher eventPublisher(KafkaTemplate<String, String> kafkaTemplate,
                                          ObjectMapper objectMapper) {
        return new KafkaEventPublisher(kafkaTemplate, objectMapper);
    }

    @Bean
    @ConditionalOnMissingBean
    public EventConsumer eventConsumer(ObjectMapper objectMapper) {
        return new KafkaEventConsumer(objectMapper);
    }
}