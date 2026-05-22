package com.metawebthree.digitaltwin;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class DigitalTwinServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(DigitalTwinServiceApplication.class, args);
    }
}