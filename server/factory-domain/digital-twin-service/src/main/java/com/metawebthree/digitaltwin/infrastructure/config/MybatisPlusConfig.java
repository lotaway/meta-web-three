package com.metawebthree.digitaltwin.infrastructure.config;

import com.baomidou.mybatisplus.annotation.DbType;
import com.metawebthree.common.config.MybatisPlusDefaultConfig;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.context.annotation.Configuration;

@Configuration
@MapperScan("com.metawebthree.digitaltwin.infrastructure.persistence.mapper")
public class MybatisPlusConfig extends MybatisPlusDefaultConfig {
    @Override
    protected DbType getInterceptorParams() {
        return DbType.POSTGRE_SQL;
    }
}
