package com.metawebthree.commission.infrastructure.config;

import java.time.LocalDateTime;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.metawebthree.commission.application.CommissionCommandService;
import com.metawebthree.commission.application.CommissionConfigProvider;
import com.metawebthree.commission.application.CommissionQueryService;
import com.metawebthree.commission.application.TimeProvider;
import com.metawebthree.commission.domain.ports.CommissionAccountStore;
import com.metawebthree.commission.domain.ports.CommissionConfigStore;
import com.metawebthree.commission.domain.ports.CommissionRecordStore;
import com.metawebthree.commission.domain.ports.CommissionRelationStore;

@Configuration
@EnableConfigurationProperties(CommissionProperties.class)
public class CommissionServiceConfig {

    @Bean
    public TimeProvider timeProvider() {
        return LocalDateTime::now;
    }

    @Bean
    public CommissionConfigProvider commissionConfigProvider(CommissionConfigStore configStore,
            CommissionProperties properties) {
        return new CommissionConfigProvider(configStore, properties);
    }

    @Bean
    public CommissionCommandService commissionCommandService(CommissionRelationStore relationStore,
            CommissionRecordStore recordStore, CommissionAccountStore accountStore,
            CommissionConfigProvider configProvider, TimeProvider timeProvider) {
        return new CommissionCommandService(relationStore, recordStore, accountStore, configProvider, timeProvider);
    }

    @Bean
    public CommissionQueryService commissionQueryService(CommissionRecordStore recordStore,
            CommissionAccountStore accountStore, CommissionConfigProvider configProvider) {
        return new CommissionQueryService(recordStore, accountStore, configProvider);
    }
}
