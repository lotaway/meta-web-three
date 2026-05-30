package com.metawebthree.reporting.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

/**
 * 报表服务配置
 */
@Configuration
public class ReportConfig {

    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}