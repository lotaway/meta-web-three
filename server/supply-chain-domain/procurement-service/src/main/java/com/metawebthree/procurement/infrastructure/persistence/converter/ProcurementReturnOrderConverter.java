package com.metawebthree.procurement.infrastructure.persistence.converter;

import com.metawebthree.procurement.application.dto.ProcurementReturnOrderDTO;
import com.metawebthree.procurement.application.dto.ProcurementReturnOrderItemDTO;
import com.metawebthree.procurement.domain.entity.ProcurementReturnOrder;
import com.metawebthree.procurement.domain.entity.ProcurementReturnOrderItem;
import com.metawebthree.procurement.infrastructure.persistence.dataobject.ProcurementReturnOrderDO;
import org.springframework.stereotype.Component;
import java.util.stream.Collectors;

@Component
public class ProcurementReturnOrderConverter {
    
    public ProcurementReturnOrder toEntity(ProcurementReturnOrderDO dto) {
        if (dto == null) return null;
        
        ProcurementReturnOrder entity = new ProcurementReturnOrder();
        entity.setId(dto.getId());
        entity.setReturnNo(dto.getReturnNo());
        entity.setSourceOrderNo(dto.getSourceOrderNo());
        entity.setSourceOrderType(dto.getSourceOrderType());
        entity.setSupplierCode(dto.getSupplierCode());
        entity.setSupplierName(dto.getSupplierName());
        entity.setWarehouseId(dto.getWarehouseId());
        entity.setWarehouseName(dto.getWarehouseName());
        entity.setReturnType(dto.getReturnType());
        entity.setStatus(dto.getStatus());
        entity.setTotalAmount(dto.getTotalAmount());
        entity.setCurrency(dto.getCurrency());
        entity.setReason(dto.getReason());
        entity.setRemark(dto.getRemark());
        entity.setApprover(dto.getApprover());
        entity.setApprovalComment(dto.getApprovalComment());
        entity.setApprovedAt(dto.getApprovedAt());
        entity.setExpectedReturnDate(dto.getExpectedReturnDate());
        entity.setActualReturnDate(dto.getActualReturnDate());
        entity.setLogisticsCompany(dto.getLogisticsCompany());
        entity.setTrackingNumber(dto.getTrackingNumber());
        entity.setShippedAt(dto.getShippedAt());
        entity.setCreatedAt(dto.getCreatedAt());
        entity.setUpdatedAt(dto.getUpdatedAt());
        return entity;
    }
    
    public ProcurementReturnOrderDO toDO(ProcurementReturnOrder entity) {
        if (entity == null) return null;
        
        ProcurementReturnOrderDO dto = new ProcurementReturnOrderDO();
        dto.setId(entity.getId());
        dto.setReturnNo(entity.getReturnNo());
        dto.setSourceOrderNo(entity.getSourceOrderNo());
        dto.setSourceOrderType(entity.getSourceOrderType());
        dto.setSupplierCode(entity.getSupplierCode());
        dto.setSupplierName(entity.getSupplierName());
        dto.setWarehouseId(entity.getWarehouseId());
        dto.setWarehouseName(entity.getWarehouseName());
        dto.setReturnType(entity.getReturnType());
        dto.setStatus(entity.getStatus());
        dto.setTotalAmount(entity.getTotalAmount());
        dto.setCurrency(entity.getCurrency());
        dto.setReason(entity.getReason());
        dto.setRemark(entity.getRemark());
        dto.setApprover(entity.getApprover());
        dto.setApprovalComment(entity.getApprovalComment());
        dto.setApprovedAt(entity.getApprovedAt());
        dto.setExpectedReturnDate(entity.getExpectedReturnDate());
        dto.setActualReturnDate(entity.getActualReturnDate());
        dto.setLogisticsCompany(entity.getLogisticsCompany());
        dto.setTrackingNumber(entity.getTrackingNumber());
        dto.setShippedAt(entity.getShippedAt());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        return dto;
    }
    
    public ProcurementReturnOrderDTO toDTO(ProcurementReturnOrder entity) {
        if (entity == null) return null;
        
        ProcurementReturnOrderDTO dto = new ProcurementReturnOrderDTO();
        dto.setId(entity.getId());
        dto.setReturnNo(entity.getReturnNo());
        dto.setSourceOrderNo(entity.getSourceOrderNo());
        dto.setSourceOrderType(entity.getSourceOrderType());
        dto.setSupplierCode(entity.getSupplierCode());
        dto.setSupplierName(entity.getSupplierName());
        dto.setWarehouseId(entity.getWarehouseId());
        dto.setWarehouseName(entity.getWarehouseName());
        dto.setReturnType(entity.getReturnType());
        dto.setStatus(entity.getStatus());
        dto.setTotalAmount(entity.getTotalAmount());
        dto.setCurrency(entity.getCurrency());
        dto.setReason(entity.getReason());
        dto.setRemark(entity.getRemark());
        dto.setApprover(entity.getApprover());
        dto.setApprovalComment(entity.getApprovalComment());
        dto.setApprovedAt(entity.getApprovedAt());
        dto.setExpectedReturnDate(entity.getExpectedReturnDate());
        dto.setActualReturnDate(entity.getActualReturnDate());
        dto.setLogisticsCompany(entity.getLogisticsCompany());
        dto.setTrackingNumber(entity.getTrackingNumber());
        dto.setShippedAt(entity.getShippedAt());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        
        if (entity.getItems() != null) {
            dto.setItems(entity.getItems().stream()
                .map(this::toItemDTO)
                .collect(Collectors.toList()));
        }
        return dto;
    }
    
    private ProcurementReturnOrderItemDTO toItemDTO(ProcurementReturnOrderItem item) {
        if (item == null) return null;
        
        ProcurementReturnOrderItemDTO dto = new ProcurementReturnOrderItemDTO();
        dto.setId(item.getId());
        dto.setReturnOrderId(item.getReturnOrderId());
        dto.setReturnNo(item.getReturnNo());
        dto.setSourceOrderNo(item.getSourceOrderNo());
        dto.setSourceOrderItemId(item.getSourceOrderItemId());
        dto.setSkuCode(item.getSkuCode());
        dto.setProductName(item.getProductName());
        dto.setReturnQuantity(item.getReturnQuantity());
        dto.setUnitPrice(item.getUnitPrice());
        dto.setTotalAmount(item.getTotalAmount());
        dto.setReason(item.getReason());
        dto.setStatus(item.getStatus());
        return dto;
    }
}