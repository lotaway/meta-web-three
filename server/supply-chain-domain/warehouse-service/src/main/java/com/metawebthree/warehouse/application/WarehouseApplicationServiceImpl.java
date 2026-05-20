package com.metawebthree.warehouse.application;

import com.metawebthree.warehouse.application.dto.WarehouseDTO;
import com.metawebthree.warehouse.application.dto.InboundOrderDTO;
import com.metawebthree.warehouse.domain.entity.Warehouse;
import com.metawebthree.warehouse.domain.entity.InboundOrder;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class WarehouseApplicationServiceImpl implements WarehouseApplicationService {

    @Override
    public WarehouseDTO createWarehouse(WarehouseDTO dto) {
        return dto;
    }

    @Override
    public WarehouseDTO updateWarehouse(Long id, WarehouseDTO dto) {
        dto.setId(id);
        return dto;
    }

    @Override
    public WarehouseDTO queryWarehouse(Long id) {
        return null;
    }

    @Override
    public List<WarehouseDTO> listWarehouses(String status) {
        return List.of();
    }

    @Override
    public InboundOrderDTO createInboundOrder(InboundOrderDTO dto) {
        return dto;
    }

    @Override
    public InboundOrderDTO confirmInboundOrder(String orderNo) {
        return null;
    }

    @Override
    public InboundOrderDTO completeInboundOrder(String orderNo, InboundOrderDTO dto) {
        return dto;
    }

    @Override
    public InboundOrderDTO queryInboundOrder(String orderNo) {
        return null;
    }

    @Override
    public List<InboundOrderDTO> listInboundOrders(Long warehouseId, String status) {
        return List.of();
    }
}