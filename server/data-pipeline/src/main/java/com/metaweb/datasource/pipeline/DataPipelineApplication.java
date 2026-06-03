package com.metaweb.datasource.pipeline;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

@SpringBootApplication
@ComponentScan(basePackages = {"com.metaweb.datasource.pipeline", "com.metaweb.common"})
public class DataPipelineApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(DataPipelineApplication.class, args);
        System.out.println("Data Pipeline Service started successfully!");
    }
}
