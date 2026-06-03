package com.metawebthree.common.cloud;

import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.cloud.client.discovery.ReactiveDiscoveryClient;
import com.alibaba.cloud.nacos.discovery.NacosServiceDiscovery;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConditionalOnBean(NacosServiceDiscovery.class)
public class FaultTolerantDiscoveryConfig {

    @Bean
    public ReactiveDiscoveryClient faultTolerantNacosDiscoveryClient(
            NacosServiceDiscovery nacosServiceDiscovery) {
        return new FaultTolerantNacosDiscoveryClient(nacosServiceDiscovery);
    }
}
