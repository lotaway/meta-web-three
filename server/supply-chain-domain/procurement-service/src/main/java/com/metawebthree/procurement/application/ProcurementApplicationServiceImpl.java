package com.metawebthree.procurement.application;

import com.metawebthree.procurement.application.dto.ProcurementOrderDTO;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class ProcurementApplicationServiceImpl implements ProcurementApplicationService {

    @Override
    public ProcurementOrderDTO createOrder(ProcurementOrderDTO dto) {
        dto.setStatus("PENDING");
        return dto;
    }

    @Override
    public ProcurementOrderDTO approveOrder(String orderNo, String approver) {
        return null;
    }

    @Override
    public ProcurementOrderDTO rejectOrder(String orderNo, String reason) {
        return null;
    }

    @Override
    public ProcurementOrderDTO queryOrder(String orderNo) {
        return null;
    }

    @Override
    public List<ProcurementOrderDTO> listOrders(String status) {
        return List.of();
    }
}