package com.metawebthree;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

import com.metawebthree.common.BaseApplication;

@SpringBootApplication
@EnableDiscoveryClient
public class GatewayApplication extends BaseApplication {

    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}
