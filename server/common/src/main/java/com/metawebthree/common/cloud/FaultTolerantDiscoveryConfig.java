package com.metawebthree.common.cloud;

import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.cloud.client.discovery.ReactiveDiscoveryClient;
import org.springframework.cloud.zookeeper.discovery.reactive.ZookeeperReactiveDiscoveryClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConditionalOnBean(ZookeeperReactiveDiscoveryClient.class)
public class FaultTolerantDiscoveryConfig {

    @Bean
    public ReactiveDiscoveryClient faultTolerantZookeeperDiscoveryClient(
            ZookeeperReactiveDiscoveryClient zookeeperReactiveDiscoveryClient) {
        return new FaultTolerantZookeeperDiscoveryClient(zookeeperReactiveDiscoveryClient);
    }
}
