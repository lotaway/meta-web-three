package com.metawebthree.common.autoconfigure;

import java.util.Map;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;

import com.metawebthree.common.event.DomainEventPublisher;

/**
 * Supplies a no-op {@link DomainEventPublisher} when no Kafka-backed publisher
 * is available, so services that publish domain events can still boot with
 * {@code meta.common.storage.kafka.enabled=false} (e.g. local/dev runs).
 */
@AutoConfiguration
public class NoOpDomainEventConfiguration {

    @Bean
    @ConditionalOnMissingBean(DomainEventPublisher.class)
    public DomainEventPublisher noOpDomainEventPublisher() {
        return new DomainEventPublisher() {
            @Override
            public void publish(String eventType, Map<String, Object> data) {
                // Intentionally no-op: Kafka storage is disabled.
            }
        };
    }
}
