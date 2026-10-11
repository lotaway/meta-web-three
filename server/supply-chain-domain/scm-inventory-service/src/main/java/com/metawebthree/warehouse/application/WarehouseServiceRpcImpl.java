package com.metawebthree.warehouse.application;

import com.metawebthree.warehouse.application.dto.WarehouseDTO;
import com.metawebthree.warehouse.application.dto.InboundOrderDTO;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class WarehouseServiceRpcImpl {

    private final WarehouseApplicationService appService;

    public WarehouseServiceRpcImpl(WarehouseApplicationService appService) {
        this.appService = appService;
    }

    public WarehouseDTO getWarehouse(Long id) {
        return appService.queryWarehouse(id);
    }

    public List<WarehouseDTO> listWarehouses(String status) {
        return appService.listWarehouses(status);
    }

    public InboundOrderDTO getInboundOrder(String orderNo) {
        return appService.queryInboundOrder(orderNo);
    }

    public InboundOrderDTO createInboundOrder(InboundOrderDTO dto) {
        return appService.createInboundOrder(dto);
    }
}