package com.metawebthree.common.config;

import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.MutablePropertySources;
import org.springframework.core.env.PropertiesPropertySource;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.beans.factory.config.YamlPropertiesFactoryBean;

import lombok.extern.slf4j.Slf4j;

import java.util.Properties;

@Slf4j
public class LocalConfig implements ApplicationContextInitializer<ConfigurableApplicationContext> {
    
    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        log.info("=== LocalConfig ApplicationContextInitializer starting ===");
        
        ConfigurableEnvironment environment = applicationContext.getEnvironment();
        String currentZkConnect = environment.getProperty("spring.cloud.zookeeper.connect-string");
        log.info("Current zookeeper connect string: {}", currentZkConnect);
        
        Resource localConfig = new ClassPathResource("application-common-local.yml");
        
        if (!localConfig.exists()) {
            localConfig = new FileSystemResource("application-common-local.yml");
            log.info("Trying to load local config from file system: {}", localConfig.getFilename());
        } else {
            log.info("Found local config in classpath: {}", localConfig.getFilename());
        }
        
        if (localConfig.exists()) {
            log.info("Loading local config file: {}", localConfig.getFilename());
            YamlPropertiesFactoryBean yamlFactory = new YamlPropertiesFactoryBean();
            yamlFactory.setResources(localConfig);
            Properties props = yamlFactory.getObject();
            
            if (props != null && !props.isEmpty()) {
                log.info("Successfully parsed YAML file with {} properties", props.size());
                props.forEach((key, value) -> {
                    if (key.toString().contains("zookeeper") || key.toString().contains("zk")) {
                        log.info("Found config: {} = {}", key, value);
                    }
                });
                
                PropertiesPropertySource localPropertySource = new PropertiesPropertySource("localConfig", props);
                MutablePropertySources propertySources = environment.getPropertySources();
                if (propertySources.contains("localConfig")) {
                    propertySources.remove("localConfig");
                    log.info("Removed existing localConfig PropertySource");
                }
                
                propertySources.addFirst(localPropertySource);
                log.info("Successfully added localConfig PropertySource to highest priority");
                String newZkConnect = environment.getProperty("spring.cloud.zookeeper.connect-string");
                log.info("Zookeeper connect string after loading local config: {}", newZkConnect);
                log.info("Current PropertySources order:");
                propertySources.forEach(ps -> {
                    if (ps.getName().contains("local") || ps.getName().contains("common")) {
                        log.info("  - {}: {}", ps.getName(), ps.getClass().getSimpleName());
                    }
                });
                
            } else {
                log.warn("YAML file parsed but no properties found");
            }
            
        } else {
            log.warn("Local config file not found in classpath or file system: application-common-local.yml");
        }
        
        log.info("=== LocalConfig ApplicationContextInitializer completed ===");
    }
}