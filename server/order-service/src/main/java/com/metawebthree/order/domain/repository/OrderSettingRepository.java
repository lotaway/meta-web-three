package com.metawebthree.order.domain.repository;

import com.metawebthree.order.domain.model.OrderSetting;

public interface OrderSettingRepository {
    void save(OrderSetting setting);
    void update(OrderSetting setting);
    OrderSetting findDefault();
}
