package com.metaweb.datasource.pipeline.repository;

import com.metaweb.datasource.pipeline.repository.entity.InventoryAnalytics;
import com.metaweb.datasource.pipeline.repository.entity.OrderAnalytics;
import com.metaweb.datasource.pipeline.repository.entity.UserBehaviorAnalytics;
import com.metaweb.datasource.pipeline.repository.mapper.InventoryAnalyticsMapper;
import com.metaweb.datasource.pipeline.repository.mapper.OrderAnalyticsMapper;
import com.metaweb.datasource.pipeline.repository.mapper.UserBehaviorAnalyticsMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

@Slf4j
@Repository
public class ClickHouseRepository {

    @Autowired
    private OrderAnalyticsMapper orderAnalyticsMapper;

    @Autowired
    private InventoryAnalyticsMapper inventoryAnalyticsMapper;

    @Autowired
    private UserBehaviorAnalyticsMapper userBehaviorAnalyticsMapper;

    public void insertOrderAnalytics(OrderAnalytics analytics) {
        orderAnalyticsMapper.insert(analytics);
    }

    public void insertInventoryAnalytics(InventoryAnalytics analytics) {
        inventoryAnalyticsMapper.insert(analytics);
    }

    public void insertUserBehaviorAnalytics(UserBehaviorAnalytics analytics) {
        userBehaviorAnalyticsMapper.insert(analytics);
    }
}
