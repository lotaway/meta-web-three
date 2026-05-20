package com.metawebthree.logistics.application;

import com.metawebthree.logistics.application.dto.LogisticsOrderDTO;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class LogisticsServiceRpcImpl {

    private final LogisticsApplicationService appService;

    public LogisticsServiceRpcImpl(LogisticsApplicationService appService) {
        this.appService = appService;
    }

    public LogisticsOrderDTO getByTrackingNo(String trackingNo) {
        return appService.queryByTrackingNo(trackingNo);
    }

    public LogisticsOrderDTO getByOrderNo(String orderNo) {
        return appService.queryByOrderNo(orderNo);
    }

    public LogisticsOrderDTO create(LogisticsOrderDTO dto) {
        return appService.createOrder(dto);
    }
}