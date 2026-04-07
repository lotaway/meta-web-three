package com.metawebthree.common.cloud;

import java.util.Collections;
import java.util.List;

import org.springframework.cloud.client.ServiceInstance;
import org.springframework.cloud.client.discovery.DiscoveryClient;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public final class DiscoveryClientSupport {

    private DiscoveryClientSupport() {
    }

    public static List<String> getServicesSafely(DiscoveryClient discoveryClient, String clientName) {
        try {
            return discoveryClient.getServices();
        } catch (Exception ex) {
            log.warn("Failed to query services from discovery client '{}': {}. Returning empty list.",
                    clientName, ex.getMessage());
            return Collections.emptyList();
        }
    }

    public static List<ServiceInstance> getInstancesSafely(
            DiscoveryClient discoveryClient,
            String serviceId,
            String clientName) {
        try {
            return discoveryClient.getInstances(serviceId);
        } catch (Exception ex) {
            log.warn("Failed to query instances for service '{}' from discovery client '{}': {}. Returning empty list.",
                    serviceId, clientName, ex.getMessage());
            return Collections.emptyList();
        }
    }
}
