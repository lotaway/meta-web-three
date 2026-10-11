package com.metawebthree.procurement.application;

import com.metawebthree.procurement.application.dto.ProcurementReturnOrderDTO;
import java.util.List;

public interface ProcurementReturnService {
    
    ProcurementReturnOrderDTO createReturnOrder(ProcurementReturnOrderDTO dto);
    
    ProcurementReturnOrderDTO submitForApproval(String returnNo);
    
    ProcurementReturnOrderDTO approveReturnOrder(String returnNo, String approver, String comment);
    
    ProcurementReturnOrderDTO rejectReturnOrder(String returnNo, String approver, String reason);
    
    ProcurementReturnOrderDTO shipReturnOrder(String returnNo, String logisticsCompany, String trackingNumber);
    
    ProcurementReturnOrderDTO confirmReturned(String returnNo);
    
    ProcurementReturnOrderDTO completeReturnOrder(String returnNo);
    
    ProcurementReturnOrderDTO cancelReturnOrder(String returnNo);
    
    ProcurementReturnOrderDTO queryReturnOrder(String returnNo);
    
    List<ProcurementReturnOrderDTO> listReturnOrders(String status, Long warehouseId, String supplierCode);
}