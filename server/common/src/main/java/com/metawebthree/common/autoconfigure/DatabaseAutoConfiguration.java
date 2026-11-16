package com.metawebthree.common.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;

import com.baomidou.mybatisplus.core.config.GlobalConfig;

/**
 * Shared MyBatis-Plus defaults. A service opts in explicitly because not every
 * service owns a datasource.
 */
@AutoConfiguration
@ConditionalOnClass(GlobalConfig.class)
@ConditionalOnProperty(prefix = "meta.common.storage.database", name = "enabled", havingValue = "true")
@EnableConfigurationProperties(CommonStorageProperties.class)
@Import(DatabaseAutoConfiguration.MybatisDefaults.class)
public class DatabaseAutoConfiguration {

    @Configuration(proxyBeanMethods = false)
    static class MybatisDefaults {
        @Bean
        @ConditionalOnMissingBean
        GlobalConfig globalConfig() {
            GlobalConfig conf = new GlobalConfig();
            conf.setDbConfig(new GlobalConfig.DbConfig().setTableUnderline(true));
            return conf;
        }
    }
}
