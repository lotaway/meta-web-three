package com.metawebthree.common.cloud;

import lombok.extern.slf4j.Slf4j;
import org.springframework.cloud.client.ServiceInstance;
import org.springframework.cloud.client.discovery.ReactiveDiscoveryClient;
import org.springframework.cloud.zookeeper.discovery.reactive.ZookeeperReactiveDiscoveryClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;

@Slf4j
public class FaultTolerantZookeeperDiscoveryClient implements ReactiveDiscoveryClient {

    private final ZookeeperReactiveDiscoveryClient delegate;

    public FaultTolerantZookeeperDiscoveryClient(ZookeeperReactiveDiscoveryClient delegate) {
        this.delegate = delegate;
    }

    @Override
    public String description() {
        return "Fault-Tolerant Zookeeper Reactive Discovery Client";
    }

    @Override
    public Flux<ServiceInstance> getInstances(String serviceId) {
        return delegate.getInstances(serviceId)
                .onErrorResume(throwable -> {
                    log.warn("Error getting instances for service '{}': {}. Returning empty list.",
                            serviceId, throwable.getMessage());
                    return Flux.empty();
                });
    }

    @Override
    public Flux<String> getServices() {
        return delegate.getServices()
                .onErrorResume(throwable -> {
                    log.warn("Error getting services list: {}. Returning empty list.", throwable.getMessage());
                    return Flux.empty();
                });
    }
}
