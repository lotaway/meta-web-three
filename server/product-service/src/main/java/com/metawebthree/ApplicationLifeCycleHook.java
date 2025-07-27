package com.metawebthree;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.ApplicationListener;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

@Component
public class ApplicationLifeCycleHook {

    @Value("${rocketmq.client.namesrv:未配置}")
    private String namesrv;

    @Value("${server.port:8080}")
    private String serverPort;

    @PostConstruct
    public void init() {
        System.out.println("=== ApplicationLifeCycleHook @PostConstruct ===");
        System.out.println("namesrv: " + namesrv);
        System.out.println("server.port: " + serverPort);
    }

    @Component
    public static class ApplicationHook implements ApplicationListener<ApplicationReadyEvent> {
        
        @Override
        public void onApplicationEvent(ApplicationReadyEvent event) {
            System.out.println("=== ApplicationLifeCycleHook ApplicationReadyEvent ===");
        }
    }

    @Component
    public static class CommandLineHook implements CommandLineRunner {
        
        @Override
        public void run(String... args) throws Exception {
            System.out.println("=== ApplicationLifeCycleHook CommandLineRunner ===");
        }
    }
}