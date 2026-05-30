package com.metawebthree.warehouse.application;

import com.metawebthree.warehouse.application.dto.QualityInspectionDTO;
import com.metawebthree.warehouse.application.dto.QualityStandardDTO;
import com.metawebthree.warehouse.domain.entity.QualityInspection;
import com.metawebthree.warehouse.domain.entity.QualityStandard;
import com.metawebthree.warehouse.infrastructure.persistence.repository.QualityInspectionRepository;
import com.metawebthree.warehouse.infrastructure.persistence.repository.QualityStandardRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class QualityInspectionApplicationService {
    
    @Autowired
    private QualityStandardRepository standardRepository;
    
    @Autowired
    private QualityInspectionRepository inspectionRepository;
    
    // Quality Standard operations
    @Transactional
    public QualityStandardDTO createStandard(QualityStandardDTO dto) {
        QualityStandard entity = new QualityStandard();
        entity.setSkuCode(dto.getSkuCode());
        entity.setProductName(dto.getProductName());
        entity.setInspectionType(dto.getInspectionType());
        entity.setInspectionLevel(dto.getInspectionLevel());
        entity.setSampleRate(dto.getSampleRate());
        entity.setCheckItems(dto.getCheckItems());
        entity.setAcceptanceQty(dto.getAcceptanceQty());
        entity.setDefectQtyThreshold(dto.getDefectQtyThreshold());
        entity.setWeightTolerance(dto.getWeightTolerance());
        entity.setDimensionTolerance(dto.getDimensionTolerance());
        entity.setPackagingRequirement(dto.getPackagingRequirement());
        entity.setLabelRequirement(dto.getLabelRequirement());
        entity.setIsActive(dto.getIsActive());
        entity.setRemark(dto.getRemark());
        entity.setCreator("system");
        entity.setCreateTime(LocalDateTime.now());
        entity.setDeleted(false);
        
        QualityStandard saved = standardRepository.save(entity);
        return toStandardDTO(saved);
    }
    
    public QualityStandardDTO getStandardBySku(String skuCode) {
        QualityStandard entity = standardRepository.findBySkuCode(skuCode);
        return entity != null ? toStandardDTO(entity) : null;
    }
    
    public List<QualityStandardDTO> listStandards(Boolean activeOnly) {
        List<QualityStandard> list;
        if (activeOnly != null && activeOnly) {
            list = standardRepository.findByActive(true);
        } else {
            list = standardRepository.findAll();
        }
        return list.stream().map(this::toStandardDTO).collect(Collectors.toList());
    }
    
    // Quality Inspection operations
    @Transactional
    public QualityInspectionDTO createInspection(QualityInspectionDTO dto) {
        QualityInspection entity = new QualityInspection();
        entity.setInspectionNo("QI-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase());
        entity.setOrderId(dto.getOrderId());
        entity.setOrderNo(dto.getOrderNo());
        entity.setInboundType(dto.getInboundType());
        entity.setWarehouseId(dto.getWarehouseId());
        entity.setSupplierCode(dto.getSupplierCode());
        entity.setSupplierName(dto.getSupplierName());
        entity.setInspectionType(dto.getInspectionType());
        entity.setInspectionStatus(QualityInspection.STATUS_PENDING);
        entity.setTotalQuantity(dto.getTotalQuantity());
        entity.setInspectedQuantity(0);
        entity.setQualifiedQuantity(0);
        entity.setUnqualifiedQuantity(0);
        entity.setConcessionQuantity(0);
        entity.setDefectRate(java.math.BigDecimal.ZERO);
        entity.setCreator("system");
        entity.setCreateTime(LocalDateTime.now());
        entity.setDeleted(false);
        
        QualityInspection saved = inspectionRepository.save(entity);
        return toInspectionDTO(saved);
    }
    
    public QualityInspectionDTO getInspectionById(Long id) {
        QualityInspection entity = inspectionRepository.findById(id);
        return entity != null ? toInspectionDTO(entity) : null;
    }
    
    public QualityInspectionDTO getInspectionByNo(String inspectionNo) {
        QualityInspection entity = inspectionRepository.findByInspectionNo(inspectionNo);
        return entity != null ? toInspectionDTO(entity) : null;
    }
    
    public List<QualityInspectionDTO> listInspections() {
        List<QualityInspection> list = inspectionRepository.findAll();
        return list.stream().map(this::toInspectionDTO).collect(Collectors.toList());
    }
    
    // DTO converters
    private QualityStandardDTO toStandardDTO(QualityStandard entity) {
        QualityStandardDTO dto = new QualityStandardDTO();
        dto.setId(entity.getId());
        dto.setSkuCode(entity.getSkuCode());
        dto.setProductName(entity.getProductName());
        dto.setInspectionType(entity.getInspectionType());
        dto.setInspectionLevel(entity.getInspectionLevel());
        dto.setSampleRate(entity.getSampleRate());
        dto.setCheckItems(entity.getCheckItems());
        dto.setAcceptanceQty(entity.getAcceptanceQty());
        dto.setDefectQtyThreshold(entity.getDefectQtyThreshold());
        dto.setWeightTolerance(entity.getWeightTolerance());
        dto.setDimensionTolerance(entity.getDimensionTolerance());
        dto.setPackagingRequirement(entity.getPackagingRequirement());
        dto.setLabelRequirement(entity.getLabelRequirement());
        dto.setIsActive(entity.getIsActive());
        dto.setRemark(entity.getRemark());
        return dto;
    }
    
    private QualityInspectionDTO toInspectionDTO(QualityInspection entity) {
        QualityInspectionDTO dto = new QualityInspectionDTO();
        dto.setId(entity.getId());
        dto.setInspectionNo(entity.getInspectionNo());
        dto.setOrderId(entity.getOrderId());
        dto.setOrderNo(entity.getOrderNo());
        dto.setInboundType(entity.getInboundType());
        dto.setWarehouseId(entity.getWarehouseId());
        dto.setSupplierCode(entity.getSupplierCode());
        dto.setSupplierName(entity.getSupplierName());
        dto.setInspectionType(entity.getInspectionType());
        dto.setInspectionStatus(entity.getInspectionStatus());
        dto.setTotalQuantity(entity.getTotalQuantity());
        dto.setInspectedQuantity(entity.getInspectedQuantity());
        dto.setQualifiedQuantity(entity.getQualifiedQuantity());
        dto.setUnqualifiedQuantity(entity.getUnqualifiedQuantity());
        dto.setConcessionQuantity(entity.getConcessionQuantity());
        dto.setDefectRate(entity.getDefectRate());
        dto.setInspector(entity.getInspector());
        dto.setInspectionTime(entity.getInspectionTime());
        dto.setResultRemark(entity.getResultRemark());
        dto.setIsAutoInspection(entity.getIsAutoInspection());
        dto.setSourceSystem(entity.getSourceSystem());
        dto.setCreateTime(entity.getCreateTime());
        return dto;
    }
}