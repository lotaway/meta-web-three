package com.metawebthree.inventoryalert.interfaces.rpc;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.inventoryalert.application.InventoryAlertQueryService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

/**
 * Dubbo RPC service implementation for inventory alert queries
 * Exposes inventory alert statistics to other microservices
 */
@Slf4j
@DubboService
@Component
@RequiredArgsConstructor
public class InventoryAlertRpcService implements InventoryAlertService {

    private final InventoryAlertQueryService queryService;

    @Override
    public GetLowStockAlertsCountResponse getLowStockAlertsCount(GetLowStockAlertsCountRequest request) {
        log.info("Dubbo call: getLowStockAlertsCount");
        try {
            Long count = queryService.getLowStockAlertsCount();
            return GetLowStockAlertsCountResponse.newBuilder()
                    .setCount(count)
                    .build();
        } catch (Exception e) {
            log.error("Failed to get low stock alerts count", e);
            return GetLowStockAlertsCountResponse.newBuilder()
                    .setCount(0L)
                    .build();
        }
    }

    @Override
    public GetAlertStatisticsResponse getAlertStatistics(GetAlertStatisticsRequest request) {
        log.info("Dubbo call: getAlertStatistics");
        try {
            Map<String, Long> distribution = queryService.getAlertStatusDistribution();
            Map<String, Long> protoDistribution = new HashMap<>();
            for (Map.Entry<String, Long> entry : distribution.entrySet()) {
                protoDistribution.put(entry.getKey(), entry.getValue());
            }
            return GetAlertStatisticsResponse.newBuilder()
                    .putAllDistribution(protoDistribution)
                    .build();
        } catch (Exception e) {
            log.error("Failed to get alert statistics", e);
            return GetAlertStatisticsResponse.newBuilder()
                    .putAllDistribution(new HashMap<>())
                    .build();
        }
    }
}