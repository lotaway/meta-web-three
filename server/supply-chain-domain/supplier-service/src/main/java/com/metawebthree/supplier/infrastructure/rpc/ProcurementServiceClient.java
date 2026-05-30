package com.metawebthree.supplier.infrastructure.rpc;

import com.metawebthree.supplier.infrastructure.rpc.dto.ProcurementOrderDTO;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.List;

/**
 * 采购服务 Feign 客户端
 * 用于供应商门户查询采购订单
 */
@FeignClient(name = "procurement-service", path = "/api/procurement")
public interface ProcurementServiceClient {

    /**
     * 查询供应商的所有订单
     */
    @GetMapping("/orders")
    List<ProcurementOrderDTO> listOrders(@RequestParam(required = false) String status);

    /**
     * 根据订单号查询订单详情
     */
    @GetMapping("/orders/{orderNo}")
    ProcurementOrderDTO queryOrder(@PathVariable("orderNo") String orderNo);
}