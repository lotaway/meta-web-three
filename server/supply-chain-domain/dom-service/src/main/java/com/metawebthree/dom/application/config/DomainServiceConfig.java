package com.metawebthree.dom.application.config;

import com.metawebthree.dom.domain.service.DomDomainService;
import com.metawebthree.dom.domain.service.DomDomainServiceImpl;
import com.metawebthree.dom.domain.repository.*;
import com.metawebthree.dom.domain.service.InventoryServiceClient;
import com.metawebthree.dom.domain.service.WarehouseServiceClient;
import com.metawebthree.dom.domain.service.DomDomainEventPublisher;
import com.metawebthree.dom.domain.service.DomSequenceGenerator;
import com.metawebthree.dom.domain.service.DomSourcingProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

@Configuration
public class DomainServiceConfig {

    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }

    @Bean
    @ConfigurationProperties(prefix = "dom.sourcing")
    public DomSourcingProperties domSourcingProperties() {
        return new DomSourcingProperties();
    }

    @Bean
    public DomDomainService domDomainService(
            DomOrderRepository domOrderRepository,
            DomOrderLineRepository domOrderLineRepository,
            FulfillmentPlanRepository fulfillmentPlanRepository,
            InventoryServiceClient inventoryServiceClient,
            WarehouseServiceClient warehouseServiceClient,
            DomDomainEventPublisher eventPublisher,
            DomSequenceGenerator sequenceGenerator,
            DomSourcingProperties sourcingProperties) {
        return new DomDomainServiceImpl(
                domOrderRepository, domOrderLineRepository,
                fulfillmentPlanRepository, inventoryServiceClient,
                warehouseServiceClient, eventPublisher,
                sequenceGenerator, sourcingProperties);
    }
}
