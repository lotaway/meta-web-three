package com.metawebthree.aiwarehouse.domain.service;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;

public interface IAIWarehouseDomainService {
    
    AICapability registerCapability(
        String capabilityId,
        String capabilityName,
        AICapability.AICapabilityType capabilityType,
        String endpointUrl,
        AICapability.FallbackType fallbackType,
        String fallbackConfig
    );
}