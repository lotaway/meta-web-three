package com.metawebthree;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

import com.metawebthree.common.BaseApplication;

@SpringBootApplication
@EnableDiscoveryClient
@EnableAsync
@EnableScheduling
public class PaymentServiceApplication extends BaseApplication {
    public static void main(String[] args) {
        SpringApplication.run(PaymentServiceApplication.class, args);
    }
} 