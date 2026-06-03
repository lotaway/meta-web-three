package com.metawebthree.common.cloud;

import lombok.extern.slf4j.Slf4j;
import org.springframework.cloud.client.ServiceInstance;
import org.springframework.cloud.client.discovery.ReactiveDiscoveryClient;
import com.alibaba.cloud.nacos.discovery.NacosServiceDiscovery;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;

/**
 * Nacos 容错发现客户端
 * 当 Nacos 服务发现失败时，返回空列表而不是抛出异常
 */
@Slf4j
public class FaultTolerantNacosDiscoveryClient implements ReactiveDiscoveryClient {

    private final NacosServiceDiscovery nacosServiceDiscovery;

    public FaultTolerantNacosDiscoveryClient(NacosServiceDiscovery nacosServiceDiscovery) {
        this.nacosServiceDiscovery = nacosServiceDiscovery;
    }

    @Override
    public String description() {
        return "Fault-Tolerant Nacos Reactive Discovery Client";
    }

    @Override
    public Flux<ServiceInstance> getInstances(String serviceId) {
        try {
            return Flux.fromIterable(nacosServiceDiscovery.getInstances(serviceId))
                    .onErrorResume(throwable -> {
                        log.warn("Error getting instances for service '{}': {}. Returning empty list.",
                                serviceId, throwable.getMessage());
                        return Flux.empty();
                    });
        } catch (Exception e) {
            log.warn("Exception getting instances for service '{}': {}. Returning empty list.",
                    serviceId, e.getMessage());
            return Flux.empty();
        }
    }

    @Override
    public Flux<String> getServices() {
        try {
            return Flux.fromIterable(nacosServiceDiscovery.getServices())
                    .onErrorResume(throwable -> {
                        log.warn("Error getting services list: {}. Returning empty list.", throwable.getMessage());
                        return Flux.empty();
                    });
        } catch (Exception e) {
            log.warn("Exception getting services list: {}. Returning empty list.", e.getMessage());
            return Flux.empty();
        }
    }
}
