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

    @Value("${rocketmq.client.namesrv:${ROCKETMQ_NAMESRV_DEFAULT}}")
    private String namesrv;

    @Value("${server.port:${SERVER_PORT_DEFAULT}}")
    private String serverPort;

    @PostConstruct
    public void init() {
        log.info("Application starting - namesrv: {}, port: {}", namesrv, serverPort);
    }
}

@Slf4j
@Component
class ApplicationHook implements ApplicationListener<ApplicationReadyEvent> {

    @Override
    public void onApplicationEvent(ApplicationReadyEvent event) {
        log.info("Application ready event triggered");
    }
}

@Slf4j
@Component
class CommandLineHook implements CommandLineRunner {

    @Override
    public void run(String... args) throws Exception {
        log.info("Command line runner executed");
    }
}