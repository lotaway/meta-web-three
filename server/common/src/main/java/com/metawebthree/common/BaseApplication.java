package com.metawebthree.common;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.client.discovery.DiscoveryClient;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public abstract class BaseApplication {

    @Autowired
    private DiscoveryClient discoveryClient;

    @Value("${spring.application.name}")
    private String name;

    @Value("${spring.cloud.zookeeper.discovery.enabled}")
    private boolean zkEnabled;

    @PostConstruct
    public void logServices() {
        log.info("After {} started, ZK discovery is {}, available services: {}", name, zkEnabled, discoveryClient.getServices());
    }
}
