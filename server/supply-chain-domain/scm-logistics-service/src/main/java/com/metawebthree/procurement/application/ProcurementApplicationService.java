package com.metawebthree.procurement.application;

import com.metawebthree.procurement.application.dto.ProcurementOrderDTO;
import java.util.List;

public interface ProcurementApplicationService {

    ProcurementOrderDTO createOrder(ProcurementOrderDTO dto);

    ProcurementOrderDTO approveOrder(String orderNo, String approver);

    ProcurementOrderDTO rejectOrder(String orderNo, String reason);

    ProcurementOrderDTO queryOrder(String orderNo);

    List<ProcurementOrderDTO> listOrders(String status);
}