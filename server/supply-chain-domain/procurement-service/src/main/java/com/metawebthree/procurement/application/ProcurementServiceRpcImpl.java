package com.metawebthree.procurement.application;

import com.metawebthree.procurement.application.dto.ProcurementOrderDTO;
import org.springframework.stereotype.Service;

@Service
public class ProcurementServiceRpcImpl {

    private final ProcurementApplicationService appService;

    public ProcurementServiceRpcImpl(ProcurementApplicationService appService) {
        this.appService = appService;
    }

    public ProcurementOrderDTO getOrder(String orderNo) {
        return appService.queryOrder(orderNo);
    }

    public ProcurementOrderDTO create(ProcurementOrderDTO dto) {
        return appService.createOrder(dto);
    }
}