package com.metawebthree.common.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.support.PropertySourcesPlaceholderConfigurer;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

@Configuration
public class LocalConfig {
    @Bean
    public PropertySourcesPlaceholderConfigurer propertySources() {
        PropertySourcesPlaceholderConfigurer configurer = new PropertySourcesPlaceholderConfigurer();
        Resource localConfig = new FileSystemResource("application-common-local.yml");
        if (localConfig.exists()) {
            configurer.setLocations(localConfig);
        }
        return configurer;
    }
}