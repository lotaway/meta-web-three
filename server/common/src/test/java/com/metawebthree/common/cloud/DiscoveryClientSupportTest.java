package com.metawebthree.common.cloud;

import java.util.List;

import org.apache.zookeeper.KeeperException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.cloud.client.ServiceInstance;
import org.springframework.cloud.client.discovery.DiscoveryClient;

public class DiscoveryClientSupportTest {

    @Test
    public void shouldReturnEmptyServicesWhenRootNodeDoesNotExist() {
        DiscoveryClient discoveryClient = Mockito.mock(DiscoveryClient.class);
        KeeperException noNode = KeeperException.create(KeeperException.Code.NONODE, "/dev/metawebthree/spring-cloud");
        Mockito.when(discoveryClient.getServices())
                .thenThrow(new IllegalStateException("Discovery lookup failed", noNode));

        List<String> services = DiscoveryClientSupport.getServicesSafely(discoveryClient, "test-client");

        Assertions.assertTrue(services.isEmpty());
    }

    @Test
    public void shouldReturnEmptyInstancesWhenRootNodeDoesNotExist() {
        DiscoveryClient discoveryClient = Mockito.mock(DiscoveryClient.class);
        KeeperException noNode = KeeperException.create(KeeperException.Code.NONODE, "/dev/metawebthree/spring-cloud");
        Mockito.when(discoveryClient.getInstances("commission-service"))
                .thenThrow(new IllegalStateException("Discovery lookup failed", noNode));

        List<ServiceInstance> instances = DiscoveryClientSupport.getInstancesSafely(
                discoveryClient,
                "commission-service",
                "test-client");

        Assertions.assertTrue(instances.isEmpty());
    }
}
