package com.metawebthree.config;

import com.metawebthree.common.config.MultiTenantMybatisConfig;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
@MapperScan("com.metawebthree.payment")
public class MybatisPlusConfig extends MultiTenantMybatisConfig {

    @Override
    protected List<String> getTenantTables() {
        return List.of("Exchange_Orders", "User_Kyc", "Credit_Profile", "payment_reconciliation_diff");
    }
}