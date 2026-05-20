package com.metawebthree.warehouse.application;

import com.metawebthree.warehouse.application.dto.WarehouseDTO;
import com.metawebthree.warehouse.application.dto.InboundOrderDTO;
import java.util.List;

public interface WarehouseApplicationService {

    WarehouseDTO createWarehouse(WarehouseDTO dto);

    WarehouseDTO updateWarehouse(Long id, WarehouseDTO dto);

    WarehouseDTO queryWarehouse(Long id);

    List<WarehouseDTO> listWarehouses(String status);

    InboundOrderDTO createInboundOrder(InboundOrderDTO dto);

    InboundOrderDTO confirmInboundOrder(String orderNo);

    InboundOrderDTO completeInboundOrder(String orderNo, InboundOrderDTO dto);

    InboundOrderDTO queryInboundOrder(String orderNo);

    List<InboundOrderDTO> listInboundOrders(Long warehouseId, String status);
}