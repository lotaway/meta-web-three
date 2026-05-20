package com.metawebthree.logistics.application;

import com.metawebthree.logistics.application.dto.LogisticsOrderDTO;
import java.util.List;

public interface LogisticsApplicationService {

    LogisticsOrderDTO createOrder(LogisticsOrderDTO dto);

    LogisticsOrderDTO queryByTrackingNo(String trackingNo);

    LogisticsOrderDTO queryByOrderNo(String orderNo);

    LogisticsOrderDTO updateStatus(String trackingNo, String status);

    List<LogisticsOrderDTO> listOrders(Long carrierId, String status);
}