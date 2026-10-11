package com.metawebthree.config;

import com.metawebthree.common.config.MultiTenantMybatisConfig;

import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class MybatisPlusConfig extends MultiTenantMybatisConfig {

    @Override
    protected List<String> getTenantTables() {
        return List.of(
                // order-service
                "tb_order", "tb_order_item", "tb_order_return_apply", "tb_company_address",
                // payment-service
                "Exchange_Orders", "User_Kyc", "Credit_Profile", "payment_reconciliation_diff",
                // cart-service
                "oms_cart_item",
                // promotion-service
                "tb_coupon", "tb_coupon_history", "tb_flash_promotion",
                "tb_flash_promotion_session", "tb_flash_promotion_product_relation",
                // after-sale-service
                "after_sale_order"
        );
    }
}