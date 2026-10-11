package com.metawebthree.supplier.application;

import com.metawebthree.supplier.application.dto.SupplierPortalOrderDTO;
import com.metawebthree.supplier.application.dto.SupplierReconciliationDTO;
import com.metawebthree.supplier.application.dto.SupplierShipmentNoticeDTO;
import com.metawebthree.supplier.domain.entity.SupplierReconciliation;
import com.metawebthree.supplier.domain.entity.SupplierReconciliationItem;
import com.metawebthree.supplier.domain.entity.SupplierShipmentNotice;
import com.metawebthree.supplier.domain.entity.SupplierShipmentNoticeItem;
import com.metawebthree.supplier.domain.repository.SupplierReconciliationRepository;
import com.metawebthree.supplier.domain.repository.SupplierShipmentNoticeRepository;
import com.metawebthree.supplier.infrastructure.rpc.ProcurementServiceClient;
import com.metawebthree.supplier.infrastructure.rpc.dto.ProcurementOrderDTO;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class SupplierPortalApplicationServiceImpl implements SupplierPortalApplicationService {

    private final SupplierShipmentNoticeRepository shipmentNoticeRepository;
    private final SupplierReconciliationRepository reconciliationRepository;
    private final ProcurementServiceClient procurementServiceClient;

    public SupplierPortalApplicationServiceImpl(
            SupplierShipmentNoticeRepository shipmentNoticeRepository,
            SupplierReconciliationRepository reconciliationRepository,
            ProcurementServiceClient procurementServiceClient) {
        this.shipmentNoticeRepository = shipmentNoticeRepository;
        this.reconciliationRepository = reconciliationRepository;
        this.procurementServiceClient = procurementServiceClient;
    }

    // ==================== 订单查询 ====================
    @Override
    public List<SupplierPortalOrderDTO> queryOrdersBySupplier(String supplierCode, String status) {
        // 通过 RPC 调用 procurement-service 获取订单列表
        List<ProcurementOrderDTO> orders = procurementServiceClient.listOrders(status);
        // 过滤出指定供应商的订单
        return orders.stream()
                .filter(order -> supplierCode.equals(order.getSupplierCode()))
                .map(this::toSupplierPortalOrderDTO)
                .collect(Collectors.toList());
    }

    @Override
    public SupplierPortalOrderDTO queryOrderDetail(String orderNo) {
        // 通过 RPC 调用 procurement-service 获取订单详情
        ProcurementOrderDTO order = procurementServiceClient.queryOrder(orderNo);
        return order != null ? toSupplierPortalOrderDTO(order) : null;
    }

    // ==================== 发货通知 ====================
    @Override
    public SupplierShipmentNoticeDTO createShipmentNotice(SupplierShipmentNoticeDTO dto) {
        SupplierShipmentNotice notice = toShipmentNoticeEntity(dto);
        notice.setStatus("DRAFT");
        notice.setNoticeNo(generateNoticeNo());
        notice = shipmentNoticeRepository.save(notice);
        return toShipmentNoticeDTO(notice);
    }

    @Override
    public SupplierShipmentNoticeDTO updateShipmentNotice(Long id, SupplierShipmentNoticeDTO dto) {
        SupplierShipmentNotice notice = shipmentNoticeRepository.findById(id)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND, "Shipment notice not found"));
        
        if (!notice.canEdit()) {
            throw new BusinessException(ResponseStatus.ORDER_STATUS_INVALID, "Current status does not allow edit");
        }
        
        updateShipmentNoticeFromDTO(notice, dto);
        notice = shipmentNoticeRepository.save(notice);
        return toShipmentNoticeDTO(notice);
    }

    @Override
    public SupplierShipmentNoticeDTO submitShipmentNotice(Long id) {
        SupplierShipmentNotice notice = shipmentNoticeRepository.findById(id)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND, "Shipment notice not found"));
        
        if (!notice.canSubmit()) {
            throw new BusinessException(ResponseStatus.ORDER_STATUS_INVALID, "Current status does not allow submit");
        }
        
        notice.submit();
        notice = shipmentNoticeRepository.save(notice);
        return toShipmentNoticeDTO(notice);
    }

    @Override
    public SupplierShipmentNoticeDTO confirmShipmentNotice(Long id, String confirmer) {
        SupplierShipmentNotice notice = shipmentNoticeRepository.findById(id)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND, "Shipment notice not found"));
        
        if (!notice.canConfirm()) {
            throw new BusinessException(ResponseStatus.ORDER_STATUS_INVALID, "Current status does not allow confirm");
        }
        
        notice.confirm(confirmer);
        notice = shipmentNoticeRepository.save(notice);
        return toShipmentNoticeDTO(notice);
    }

    @Override
    public SupplierShipmentNoticeDTO queryShipmentNotice(Long id) {
        return shipmentNoticeRepository.findById(id)
                .map(this::toShipmentNoticeDTO)
                .orElse(null);
    }

    @Override
    public List<SupplierShipmentNoticeDTO> queryShipmentNotices(String supplierCode, String status) {
        List<SupplierShipmentNotice> notices;
        if (status != null && !status.isEmpty()) {
            notices = shipmentNoticeRepository.findBySupplierCodeAndStatus(supplierCode, status);
        } else {
            notices = shipmentNoticeRepository.findBySupplierCode(supplierCode);
        }
        return notices.stream().map(this::toShipmentNoticeDTO).collect(Collectors.toList());
    }

    // ==================== 对账 ====================
    @Override
    public SupplierReconciliationDTO createReconciliation(SupplierReconciliationDTO dto) {
        SupplierReconciliation reconciliation = toReconciliationEntity(dto);
        reconciliation.setStatus("PENDING");
        reconciliation.setReconciliationNo(generateReconciliationNo());
        reconciliation.calculateTotals();
        reconciliation = reconciliationRepository.save(reconciliation);
        return toReconciliationDTO(reconciliation);
    }

    @Override
    public SupplierReconciliationDTO submitReconciliation(Long id) {
        SupplierReconciliation reconciliation = reconciliationRepository.findById(id)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND, "Reconciliation not found"));
        
        if (!reconciliation.canSubmit()) {
            throw new BusinessException(ResponseStatus.ORDER_STATUS_INVALID, "Current status does not allow submit");
        }
        
        reconciliation.submit();
        reconciliation = reconciliationRepository.save(reconciliation);
        return toReconciliationDTO(reconciliation);
    }

    @Override
    public SupplierReconciliationDTO confirmReconciliation(Long id, String confirmedBy) {
        SupplierReconciliation reconciliation = reconciliationRepository.findById(id)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND, "Reconciliation not found"));
        
        if (!reconciliation.canConfirm()) {
            throw new BusinessException(ResponseStatus.ORDER_STATUS_INVALID, "Current status does not allow confirm");
        }
        
        reconciliation.confirm(confirmedBy);
        reconciliation = reconciliationRepository.save(reconciliation);
        return toReconciliationDTO(reconciliation);
    }

    @Override
    public SupplierReconciliationDTO rejectReconciliation(Long id, String remark) {
        SupplierReconciliation reconciliation = reconciliationRepository.findById(id)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND, "Reconciliation not found"));
        
        if (!reconciliation.canReject()) {
            throw new BusinessException(ResponseStatus.ORDER_STATUS_INVALID, "Current status does not allow reject");
        }
        
        reconciliation.reject(remark);
        reconciliation = reconciliationRepository.save(reconciliation);
        return toReconciliationDTO(reconciliation);
    }

    @Override
    public SupplierReconciliationDTO markAsPaid(Long id) {
        SupplierReconciliation reconciliation = reconciliationRepository.findById(id)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND, "Reconciliation not found"));
        
        if (!reconciliation.canMarkPaid()) {
            throw new BusinessException(ResponseStatus.ORDER_STATUS_INVALID, "Current status does not allow mark as paid");
        }
        
        reconciliation.markAsPaid();
        reconciliation = reconciliationRepository.save(reconciliation);
        return toReconciliationDTO(reconciliation);
    }

    @Override
    public SupplierReconciliationDTO queryReconciliation(Long id) {
        return reconciliationRepository.findById(id)
                .map(this::toReconciliationDTO)
                .orElse(null);
    }

    @Override
    public List<SupplierReconciliationDTO> queryReconciliations(String supplierCode, String status) {
        List<SupplierReconciliation> reconciliations;
        if (status != null && !status.isEmpty()) {
            reconciliations = reconciliationRepository.findBySupplierCodeAndStatus(supplierCode, status);
        } else {
            reconciliations = reconciliationRepository.findBySupplierCode(supplierCode);
        }
        return reconciliations.stream().map(this::toReconciliationDTO).collect(Collectors.toList());
    }

    // ==================== 辅助方法 ====================
    private String generateNoticeNo() {
        return "SN" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")) 
               + UUID.randomUUID().toString().substring(0, 8).toUpperCase();
    }

    private String generateReconciliationNo() {
        return "RC" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")) 
               + UUID.randomUUID().toString().substring(0, 8).toUpperCase();
    }

    private SupplierPortalOrderDTO toSupplierPortalOrderDTO(ProcurementOrderDTO order) {
        SupplierPortalOrderDTO dto = new SupplierPortalOrderDTO();
        dto.setOrderNo(order.getOrderNo());
        dto.setSupplierCode(order.getSupplierCode());
        dto.setPurchaseType(order.getPurchaseType());
        dto.setStatus(order.getStatus());
        dto.setTotalAmount(order.getTotalAmount());
        dto.setCurrency(order.getCurrency());
        dto.setPaymentTerms(order.getPaymentTerms());
        dto.setDeliveryTerms(order.getDeliveryTerms());
        dto.setExpectedDeliveryDate(order.getExpectedDeliveryDate());
        dto.setActualDeliveryDate(order.getActualDeliveryDate());
        dto.setRemark(order.getRemark());
        dto.setCreatedAt(order.getCreatedAt());
        return dto;
    }

    private void updateShipmentNoticeFromDTO(SupplierShipmentNotice notice, SupplierShipmentNoticeDTO dto) {
        notice.setOrderNo(dto.getOrderNo());
        notice.setWarehouseId(dto.getWarehouseId());
        notice.setExpectedShipmentDate(dto.getExpectedShipmentDate());
        notice.setShipmentMethod(dto.getShipmentMethod());
        notice.setCarrierName(dto.getCarrierName());
        notice.setCarrierContact(dto.getCarrierContact());
        notice.setTrackingNumber(dto.getTrackingNumber());
        notice.setVehicleNumber(dto.getVehicleNumber());
        notice.setDriverName(dto.getDriverName());
        notice.setDriverPhone(dto.getDriverPhone());
        notice.setTotalQuantity(dto.getTotalQuantity());
        notice.setTotalWeight(dto.getTotalWeight());
        notice.setTotalVolume(dto.getTotalVolume());
        notice.setRemark(dto.getRemark());
    }

    private SupplierShipmentNotice toShipmentNoticeEntity(SupplierShipmentNoticeDTO dto) {
        SupplierShipmentNotice notice = new SupplierShipmentNotice();
        notice.setId(dto.getId());
        notice.setNoticeNo(dto.getNoticeNo());
        notice.setSupplierCode(dto.getSupplierCode());
        notice.setOrderNo(dto.getOrderNo());
        notice.setWarehouseId(dto.getWarehouseId());
        notice.setExpectedShipmentDate(dto.getExpectedShipmentDate());
        notice.setActualShipmentDate(dto.getActualShipmentDate());
        notice.setShipmentMethod(dto.getShipmentMethod());
        notice.setCarrierName(dto.getCarrierName());
        notice.setCarrierContact(dto.getCarrierContact());
        notice.setTrackingNumber(dto.getTrackingNumber());
        notice.setVehicleNumber(dto.getVehicleNumber());
        notice.setDriverName(dto.getDriverName());
        notice.setDriverPhone(dto.getDriverPhone());
        notice.setTotalQuantity(dto.getTotalQuantity());
        notice.setTotalWeight(dto.getTotalWeight());
        notice.setTotalVolume(dto.getTotalVolume());
        notice.setStatus(dto.getStatus());
        notice.setRemark(dto.getRemark());
        notice.setCreatedAt(dto.getCreatedAt());
        notice.setUpdatedAt(dto.getUpdatedAt());
        
        if (dto.getItems() != null) {
            for (var itemDTO : dto.getItems()) {
                SupplierShipmentNoticeItem item = new SupplierShipmentNoticeItem();
                item.setId(itemDTO.getId());
                item.setProductCode(itemDTO.getProductCode());
                item.setProductName(itemDTO.getProductName());
                item.setUnit(itemDTO.getUnit());
                item.setQuantity(itemDTO.getQuantity());
                item.setWeight(itemDTO.getWeight());
                item.setVolume(itemDTO.getVolume());
                item.setBatchNo(itemDTO.getBatchNo());
                item.setProductionDate(itemDTO.getProductionDate());
                item.setExpiryDate(itemDTO.getExpiryDate());
                notice.getItems().add(item);
            }
        }
        return notice;
    }

    private SupplierShipmentNoticeDTO toShipmentNoticeDTO(SupplierShipmentNotice notice) {
        SupplierShipmentNoticeDTO dto = new SupplierShipmentNoticeDTO();
        dto.setId(notice.getId());
        dto.setNoticeNo(notice.getNoticeNo());
        dto.setSupplierCode(notice.getSupplierCode());
        dto.setOrderNo(notice.getOrderNo());
        dto.setWarehouseId(notice.getWarehouseId());
        dto.setExpectedShipmentDate(notice.getExpectedShipmentDate());
        dto.setActualShipmentDate(notice.getActualShipmentDate());
        dto.setShipmentMethod(notice.getShipmentMethod());
        dto.setCarrierName(notice.getCarrierName());
        dto.setCarrierContact(notice.getCarrierContact());
        dto.setTrackingNumber(notice.getTrackingNumber());
        dto.setVehicleNumber(notice.getVehicleNumber());
        dto.setDriverName(notice.getDriverName());
        dto.setDriverPhone(notice.getDriverPhone());
        dto.setTotalQuantity(notice.getTotalQuantity());
        dto.setTotalWeight(notice.getTotalWeight());
        dto.setTotalVolume(notice.getTotalVolume());
        dto.setStatus(notice.getStatus());
        dto.setRemark(notice.getRemark());
        dto.setConfirmer(notice.getConfirmer());
        dto.setConfirmedAt(notice.getConfirmedAt());
        dto.setCreatedAt(notice.getCreatedAt());
        dto.setUpdatedAt(notice.getUpdatedAt());
        
        if (notice.getItems() != null) {
            dto.setItems(notice.getItems().stream().map(item -> {
                var itemDTO = new SupplierShipmentNoticeDTO.SupplierShipmentNoticeItemDTO();
                itemDTO.setId(item.getId());
                itemDTO.setProductCode(item.getProductCode());
                itemDTO.setProductName(item.getProductName());
                itemDTO.setUnit(item.getUnit());
                itemDTO.setQuantity(item.getQuantity());
                itemDTO.setWeight(item.getWeight());
                itemDTO.setVolume(item.getVolume());
                itemDTO.setBatchNo(item.getBatchNo());
                itemDTO.setProductionDate(item.getProductionDate());
                itemDTO.setExpiryDate(item.getExpiryDate());
                return itemDTO;
            }).collect(Collectors.toList()));
        }
        return dto;
    }

    private SupplierReconciliation toReconciliationEntity(SupplierReconciliationDTO dto) {
        SupplierReconciliation reconciliation = new SupplierReconciliation();
        reconciliation.setId(dto.getId());
        reconciliation.setReconciliationNo(dto.getReconciliationNo());
        reconciliation.setSupplierCode(dto.getSupplierCode());
        reconciliation.setPeriodStart(dto.getPeriodStart());
        reconciliation.setPeriodEnd(dto.getPeriodEnd());
        reconciliation.setTotalAmount(dto.getTotalAmount());
        reconciliation.setShippedAmount(dto.getShippedAmount());
        reconciliation.setInvoicedAmount(dto.getInvoicedAmount());
        reconciliation.setSettledAmount(dto.getSettledAmount());
        reconciliation.setPendingAmount(dto.getPendingAmount());
        reconciliation.setCurrency(dto.getCurrency());
        reconciliation.setStatus(dto.getStatus());
        reconciliation.setRemark(dto.getRemark());
        reconciliation.setCreatedAt(dto.getCreatedAt());
        reconciliation.setUpdatedAt(dto.getUpdatedAt());
        
        if (dto.getItems() != null) {
            for (var itemDTO : dto.getItems()) {
                SupplierReconciliationItem item = new SupplierReconciliationItem();
                item.setId(itemDTO.getId());
                item.setOrderNo(itemDTO.getOrderNo());
                item.setOrderDate(itemDTO.getOrderDate());
                item.setShippedDate(itemDTO.getShippedDate());
                item.setInvoicedAmount(itemDTO.getInvoicedAmount());
                item.setSettledAmount(itemDTO.getSettledAmount());
                item.setPendingAmount(itemDTO.getPendingAmount());
                item.setStatus(itemDTO.getStatus());
                item.setRemark(itemDTO.getRemark());
                reconciliation.getItems().add(item);
            }
        }
        return reconciliation;
    }

    private SupplierReconciliationDTO toReconciliationDTO(SupplierReconciliation reconciliation) {
        SupplierReconciliationDTO dto = new SupplierReconciliationDTO();
        dto.setId(reconciliation.getId());
        dto.setReconciliationNo(reconciliation.getReconciliationNo());
        dto.setSupplierCode(reconciliation.getSupplierCode());
        dto.setPeriodStart(reconciliation.getPeriodStart());
        dto.setPeriodEnd(reconciliation.getPeriodEnd());
        dto.setOrderCount(reconciliation.getOrderCount());
        dto.setTotalAmount(reconciliation.getTotalAmount());
        dto.setShippedAmount(reconciliation.getShippedAmount());
        dto.setInvoicedAmount(reconciliation.getInvoicedAmount());
        dto.setSettledAmount(reconciliation.getSettledAmount());
        dto.setPendingAmount(reconciliation.getPendingAmount());
        dto.setCurrency(reconciliation.getCurrency());
        dto.setStatus(reconciliation.getStatus());
        dto.setSubmittedAt(reconciliation.getSubmittedAt());
        dto.setConfirmedAt(reconciliation.getConfirmedAt());
        dto.setConfirmedBy(reconciliation.getConfirmedBy());
        dto.setPaidAt(reconciliation.getPaidAt());
        dto.setRemark(reconciliation.getRemark());
        dto.setCreatedAt(reconciliation.getCreatedAt());
        dto.setUpdatedAt(reconciliation.getUpdatedAt());
        
        if (reconciliation.getItems() != null) {
            dto.setItems(reconciliation.getItems().stream().map(item -> {
                var itemDTO = new SupplierReconciliationDTO.SupplierReconciliationItemDTO();
                itemDTO.setId(item.getId());
                itemDTO.setOrderNo(item.getOrderNo());
                itemDTO.setOrderDate(item.getOrderDate());
                itemDTO.setShippedDate(item.getShippedDate());
                itemDTO.setInvoicedAmount(item.getInvoicedAmount());
                itemDTO.setSettledAmount(item.getSettledAmount());
                itemDTO.setPendingAmount(item.getPendingAmount());
                itemDTO.setStatus(item.getStatus());
                itemDTO.setRemark(item.getRemark());
                return itemDTO;
            }).collect(Collectors.toList()));
        }
        return dto;
    }
}