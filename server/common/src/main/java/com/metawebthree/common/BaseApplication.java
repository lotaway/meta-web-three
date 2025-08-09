package com.metawebthree.common;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.client.discovery.DiscoveryClient;
import org.springframework.context.annotation.EnableAspectJAutoProxy;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@EnableAspectJAutoProxy
public abstract class BaseApplication {

    @Autowired
    private DiscoveryClient discoveryClient;

    @Value("${spring.application.name}")
    private String name;

    @Value("${spring.cloud.zookeeper.connect-string}")
    private String zkConnected;

    @PostConstruct
    public void logServices() {
        log.info(
                "After {} started, ZK connection is {}, available services: {}",
                name,
                zkConnected,
                discoveryClient.getServices());
    }
}
