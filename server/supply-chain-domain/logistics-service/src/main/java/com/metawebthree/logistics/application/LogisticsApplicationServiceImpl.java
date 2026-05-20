package com.metawebthree.logistics.application;

import com.metawebthree.logistics.application.dto.LogisticsOrderDTO;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class LogisticsApplicationServiceImpl implements LogisticsApplicationService {

    @Override
    public LogisticsOrderDTO createOrder(LogisticsOrderDTO dto) {
        dto.setStatus("CREATED");
        return dto;
    }

    @Override
    public LogisticsOrderDTO queryByTrackingNo(String trackingNo) {
        return null;
    }

    @Override
    public LogisticsOrderDTO queryByOrderNo(String orderNo) {
        return null;
    }

    @Override
    public LogisticsOrderDTO updateStatus(String trackingNo, String status) {
        return null;
    }

    @Override
    public List<LogisticsOrderDTO> listOrders(Long carrierId, String status) {
        return List.of();
    }
}