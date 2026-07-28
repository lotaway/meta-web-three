package com.metawebthree.config;

import com.metawebthree.common.config.MultiTenantMybatisConfig;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
@MapperScan("com.metawebthree.cart")
public class MybatisPlusConfig extends MultiTenantMybatisConfig {

    @Override
    protected List<String> getTenantTables() {
        return List.of("oms_cart_item");
    }
}
