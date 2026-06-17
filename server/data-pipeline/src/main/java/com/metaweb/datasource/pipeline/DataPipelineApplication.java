package com.metaweb.datasource.pipeline;

import com.baomidou.mybatisplus.autoconfigure.MybatisPlusAutoConfiguration;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.scheduling.annotation.EnableScheduling;

@Slf4j
@SpringBootApplication(exclude = MybatisPlusAutoConfiguration.class)
@ComponentScan(basePackages = {"com.metaweb.datasource.pipeline", "com.metaweb.common"})
@EnableScheduling
public class DataPipelineApplication {

    public static void main(String[] args) {
        SpringApplication.run(DataPipelineApplication.class, args);
        log.info("Data Pipeline Service started successfully!");
    }
}
