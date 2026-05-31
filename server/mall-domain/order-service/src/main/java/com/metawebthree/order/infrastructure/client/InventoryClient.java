package com.metawebthree.order.infrastructure.client;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.supplychain.generated.rpc.InventoryService;
import com.metawebthree.supplychain.generated.rpc.ReleaseInventoryRequest;
import com.metawebthree.supplychain.generated.rpc.ReleaseInventoryResponse;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class InventoryClient {

    @DubboReference(check = false, lazy = true)
    private InventoryService inventoryService;

    /**
     * 释放订单占用的库存
     * @param reservationId 库存预留ID（从下单时返回的reservationId获取）
     * @return 是否释放成功
     */
    public boolean releaseInventory(String reservationId) {
        try {
            ReleaseInventoryRequest request = ReleaseInventoryRequest.newBuilder()
                    .setReservationId(reservationId)
                    .build();

            ReleaseInventoryResponse response = inventoryService.releaseInventory(request);
            if (response.getSuccess()) {
                log.info("库存释放成功 - reservationId: {}", reservationId);
                return true;
            } else {
                log.warn("库存释放失败 - reservationId: {}, message: {}", 
                        reservationId, response.getMessage());
                return false;
            }
        } catch (Exception e) {
            log.error("库存释放异常 - reservationId: {}, error: {}", 
                    reservationId, e.getMessage(), e);
            return false;
        }
    }
}