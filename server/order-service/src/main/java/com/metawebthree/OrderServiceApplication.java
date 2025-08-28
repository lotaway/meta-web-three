package com.metawebthree;

import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

import com.metawebthree.common.BaseApplication;

@SpringBootApplication
@EnableDiscoveryClient
@EnableDubbo
public class OrderServiceApplication extends BaseApplication {

    public static void main(String[] args) {
        SpringApplication.run(OrderServiceApplication.class, args);
    }

}
