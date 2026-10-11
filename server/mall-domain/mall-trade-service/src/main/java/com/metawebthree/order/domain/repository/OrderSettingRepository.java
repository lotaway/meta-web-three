package com.metawebthree.order.domain.repository;

import com.metawebthree.order.domain.model.OrderSetting;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface OrderSettingRepository {
    void save(OrderSetting setting);
    void update(OrderSetting setting);
    OrderSetting findDefault();
}
