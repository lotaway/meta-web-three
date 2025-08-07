package com.metawebthree.common.config;

import com.baomidou.mybatisplus.annotation.DbType;
import com.baomidou.mybatisplus.autoconfigure.ConfigurationCustomizer;
import com.baomidou.mybatisplus.core.handlers.MetaObjectHandler;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.baomidou.mybatisplus.extension.plugins.MybatisPlusInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.PaginationInnerInterceptor;

import java.sql.Timestamp;
import java.time.LocalDateTime;

import org.apache.ibatis.reflection.MetaObject;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

// extends this config and add @Configuration and @MapperScan("scan.your.mapper.package") to use pagination
@Configuration
public class MybatisPlusDefaultConfig {
    /**
     * new page plugin, one cache and two cache follow the rules of mybatis, need to set, need to set MybatisConfiguration#useDeprecatedExecutor = false to avoid cache problems (this attribute will be removed together with the old plugin)
     */
    @Bean
    public MybatisPlusInterceptor mybatisPlusInterceptor() {
        MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
        interceptor.addInnerInterceptor(new PaginationInnerInterceptor(getInterceptorParams()));
        return interceptor;
    }

    protected DbType getInterceptorParams() {
        return DbType.MYSQL;
    }

    @Bean
    public ConfigurationCustomizer configurationCustomizer() {
        return configuration -> configuration.setUseGeneratedShortKey(false);
    }

    @Bean
    public MetaObjectHandler metaObjectHandler() {
        return new MetaObjectHandler() {
            @Override
            public void insertFill(MetaObject metaObject) {
                this.strictInsertFill(metaObject, "id", Long.class, IdWorker.getId());
                
                Timestamp ts = Timestamp.valueOf(LocalDateTime.now());
                this.strictInsertFill(metaObject, "createdAt", Timestamp.class, ts);
                this.strictInsertFill(metaObject, "createTime", Timestamp.class, ts);
                this.strictInsertFill(metaObject, "updatedAt", Timestamp.class, ts);
                this.strictInsertFill(metaObject, "updateTime", Timestamp.class, ts);
            }
            
            @Override
            public void updateFill(MetaObject metaObject) {
                Timestamp ts = Timestamp.valueOf(LocalDateTime.now());
                this.strictUpdateFill(metaObject, "updatedAt", Timestamp.class, ts);
                this.strictUpdateFill(metaObject, "updateTime", Timestamp.class, ts);
            }
        };
    }
}