package com.metawebthree.common.config;

import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.baomidou.mybatisplus.core.config.GlobalConfig;

@Configuration
public class DatabaseConfig {
    @Bean
    @ConditionalOnMissingBean
    public GlobalConfig globalConfig() {
        GlobalConfig conf = new GlobalConfig();
        conf.setDbConfig(new GlobalConfig.DbConfig().setTableUnderline(true));
        return conf;
    }
}