package com.metawebthree.digitaltwin;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.scheduling.annotation.EnableScheduling;

import com.metawebthree.common.BaseApplication;

@SpringBootApplication
@EnableDiscoveryClient
@EnableDubbo
@EnableScheduling
public class DigitalTwinServiceApplication extends BaseApplication {

    public static void main(String[] args) {
        SpringApplication.run(DigitalTwinServiceApplication.class, args);
    }

}