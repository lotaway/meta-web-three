package com.metawebthree;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.ApplicationListener;
import org.springframework.stereotype.Component;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;


@Slf4j
@Component
public class ApplicationLifeCycleHook {

    @Value("${rocketmq.client.namesrv:未配置}")
    private String namesrv;

    @Value("${server.port:8080}")
    private String serverPort;

    @PostConstruct
    public void init() {
        log.info("=== ApplicationLifeCycleHook @PostConstruct ===");
        log.info("namesrv: " + namesrv);
        log.info("server.port: " + serverPort);
    }
}

@Slf4j
@Component
class ApplicationHook implements ApplicationListener<ApplicationReadyEvent> {

    @Override
    public void onApplicationEvent(ApplicationReadyEvent event) {
        log.info("=== ApplicationLifeCycleHook ApplicationReadyEvent ===");
    }
}

@Slf4j
@Component
class CommandLineHook implements CommandLineRunner {

    @Override
    public void run(String... args) throws Exception {
        log.info("=== ApplicationLifeCycleHook CommandLineRunner ===");
    }
}