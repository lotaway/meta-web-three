package com.metawebthree.mes.config;

import com.metawebthree.common.event.DomainEventPublisher;
import com.metawebthree.mes.infrastructure.event.MesEventPublisher;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.context.annotation.ComponentScan;

import java.util.Map;

/**
 * Test configuration providing mock beans for integration tests.
 */
@TestConfiguration
@ComponentScan(basePackages = {"com.metawebthree.mes"})
public class TestConfig {

    @Bean
    @Primary
    public DomainEventPublisher domainEventPublisher() {
        return new DomainEventPublisher() {
            @Override
            public void publish(String eventType, Map<String, Object> data) {
                // No-op for tests
            }
        };
    }

    @Bean
    @Primary
    public MesEventPublisher mesEventPublisher(DomainEventPublisher eventPublisher) {
        return new MesEventPublisher(eventPublisher);
    }
}