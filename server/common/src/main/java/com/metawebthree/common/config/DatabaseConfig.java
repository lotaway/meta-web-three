package com.metawebthree.common.config;

import org.springframework.context.annotation.Bean;

import com.baomidou.mybatisplus.core.config.GlobalConfig;

public class DatabaseConfig {
    @Bean
    public GlobalConfig globalConfig() {
        GlobalConfig conf = new GlobalConfig();
        conf.setDbConfig(new GlobalConfig.DbConfig().setTableUnderline(true));
        return conf;
    }
}