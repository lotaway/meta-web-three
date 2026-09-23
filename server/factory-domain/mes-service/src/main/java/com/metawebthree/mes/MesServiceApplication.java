package com.metawebthree.mes;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

@SpringBootApplication
@ComponentScan("com.metawebthree.common.event")
public class MesServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(MesServiceApplication.class, args);
    }
}