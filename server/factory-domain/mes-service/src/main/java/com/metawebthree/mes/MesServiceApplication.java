package com.metawebthree.mes;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication
@EnableDiscoveryClient
public class MesServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(MesServiceApplication.class, args);
    }
}