package com.metawebthree;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.quartz.QuartzAutoConfiguration;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.scheduling.annotation.EnableScheduling;

import com.metawebthree.common.BaseApplication;
import com.metawebthree.gateway.auth.GatewayAuthProperties;

@SpringBootApplication(exclude = QuartzAutoConfiguration.class)
@EnableDiscoveryClient
@EnableScheduling
@EnableConfigurationProperties(GatewayAuthProperties.class)
public class GatewayApplication extends BaseApplication {

    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}
