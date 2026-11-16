package com.metawebthree.common.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.autoconfigure.kafka.KafkaAutoConfiguration;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.kafka.core.KafkaTemplate;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.common.event.DomainEventPublisher;
import com.metawebthree.common.event.KafkaDomainEventPublisher;

/**
 * Supplies the Kafka-backed {@link DomainEventPublisher} to any service that
 * already has a {@link KafkaTemplate}. Previously each application had to add
 * {@code com.metawebthree.common.event} to its {@code scanBasePackages}; a
 * missing entry failed the context at startup with an unsatisfied dependency.
 */
@AutoConfiguration(after = KafkaAutoConfiguration.class)
@ConditionalOnClass({KafkaTemplate.class, DomainEventPublisher.class})
@ConditionalOnProperty(prefix = "meta.common.storage.kafka", name = "enabled", havingValue = "true")
@EnableConfigurationProperties(CommonStorageProperties.class)
public class DomainEventAutoConfiguration {

    @Bean
    @ConditionalOnBean(KafkaTemplate.class)
    @ConditionalOnMissingBean(DomainEventPublisher.class)
    public DomainEventPublisher kafkaDomainEventPublisher(
            KafkaTemplate<String, String> kafkaTemplate, ObjectMapper objectMapper) {
        return new KafkaDomainEventPublisher(kafkaTemplate, objectMapper);
    }
}
