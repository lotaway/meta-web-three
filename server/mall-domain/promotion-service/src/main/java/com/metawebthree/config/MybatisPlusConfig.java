package com.metawebthree.config;

import com.metawebthree.common.config.MultiTenantMybatisConfig;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
@MapperScan("com.metawebthree.promotion.infrastructure.persistence.mapper")
public class MybatisPlusConfig extends MultiTenantMybatisConfig {

    @Override
    protected List<String> getTenantTables() {
        return List.of("tb_coupon", "tb_coupon_history", "tb_flash_promotion",
                "tb_flash_promotion_session", "tb_flash_promotion_product_relation");
    }
}
