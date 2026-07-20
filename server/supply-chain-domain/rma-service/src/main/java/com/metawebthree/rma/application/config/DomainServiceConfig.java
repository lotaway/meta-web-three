package com.metawebthree.rma.application.config;

import com.metawebthree.rma.domain.service.RmaDomainService;
import com.metawebthree.rma.domain.service.RmaDomainServiceImpl;
import com.metawebthree.rma.domain.repository.*;
import com.metawebthree.rma.domain.RmaSequenceGenerator;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class DomainServiceConfig {

    @Bean
    public RmaDomainService rmaDomainService(
            RmaOrderRepository rmaOrderRepository,
            RmaOrderItemRepository rmaOrderItemRepository,
            RmaInspectionRepository rmaInspectionRepository,
            RmaDispositionRepository rmaDispositionRepository,
            ReturnShippingRepository returnShippingRepository,
            RmaSequenceGenerator rmaSequenceGenerator) {
        return new RmaDomainServiceImpl(
                rmaOrderRepository, rmaOrderItemRepository,
                rmaInspectionRepository, rmaDispositionRepository,
                returnShippingRepository, rmaSequenceGenerator);
    }
}
