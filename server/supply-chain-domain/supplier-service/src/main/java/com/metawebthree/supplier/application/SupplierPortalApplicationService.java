package com.metawebthree.supplier.application;

import com.metawebthree.supplier.application.dto.SupplierReconciliationDTO;
import com.metawebthree.supplier.application.dto.SupplierShipmentNoticeDTO;
import com.metawebthree.supplier.application.dto.SupplierPortalOrderDTO;
import java.util.List;

public interface SupplierPortalApplicationService {

    // 订单查询
    List<SupplierPortalOrderDTO> queryOrdersBySupplier(String supplierCode, String status);

    SupplierPortalOrderDTO queryOrderDetail(String orderNo);

    // 发货通知
    SupplierShipmentNoticeDTO createShipmentNotice(SupplierShipmentNoticeDTO dto);

    SupplierShipmentNoticeDTO updateShipmentNotice(Long id, SupplierShipmentNoticeDTO dto);

    SupplierShipmentNoticeDTO submitShipmentNotice(Long id);

    SupplierShipmentNoticeDTO confirmShipmentNotice(Long id, String confirmer);

    SupplierShipmentNoticeDTO queryShipmentNotice(Long id);

    List<SupplierShipmentNoticeDTO> queryShipmentNotices(String supplierCode, String status);

    // 对账
    SupplierReconciliationDTO createReconciliation(SupplierReconciliationDTO dto);

    SupplierReconciliationDTO submitReconciliation(Long id);

    SupplierReconciliationDTO confirmReconciliation(Long id, String confirmedBy);

    SupplierReconciliationDTO rejectReconciliation(Long id, String remark);

    SupplierReconciliationDTO markAsPaid(Long id);

    SupplierReconciliationDTO queryReconciliation(Long id);

    List<SupplierReconciliationDTO> queryReconciliations(String supplierCode, String status);
}