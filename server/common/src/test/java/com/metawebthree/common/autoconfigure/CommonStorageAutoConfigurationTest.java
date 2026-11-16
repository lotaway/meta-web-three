package com.metawebthree.common.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.kafka.core.KafkaTemplate;

import com.baomidou.mybatisplus.core.config.GlobalConfig;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.common.event.DomainEventPublisher;

/**
 * Verifies that shared storage beans stay invisible until a service opts in.
 */
class CommonStorageAutoConfigurationTest {

    private final ApplicationContextRunner runner = new ApplicationContextRunner();

    @Test
    void databaseIsDisabledByDefault() {
        runner.withConfiguration(AutoConfigurations.of(DatabaseAutoConfiguration.class))
                .run(context -> assertThat(context).doesNotHaveBean(GlobalConfig.class));
    }

    @Test
    void databaseIsEnabledExplicitly() {
        runner.withPropertyValues("meta.common.storage.database.enabled=true")
                .withConfiguration(AutoConfigurations.of(DatabaseAutoConfiguration.class))
                .run(context -> assertThat(context).hasSingleBean(GlobalConfig.class));
    }

    @Test
    void domainEventsRequireBothKafkaTemplateAndOptIn() {
        runner.withBean(KafkaTemplate.class, () -> mock(KafkaTemplate.class))
                .withBean(ObjectMapper.class, ObjectMapper::new)
                .withConfiguration(AutoConfigurations.of(DomainEventAutoConfiguration.class))
                .run(context -> assertThat(context).doesNotHaveBean(DomainEventPublisher.class));

        runner.withPropertyValues("meta.common.storage.kafka.enabled=true")
                .withBean(KafkaTemplate.class, () -> mock(KafkaTemplate.class))
                .withBean(ObjectMapper.class, ObjectMapper::new)
                .withConfiguration(AutoConfigurations.of(DomainEventAutoConfiguration.class))
                .run(context -> assertThat(context).hasSingleBean(DomainEventPublisher.class));
    }
}
