package com.metawebthree.supplier.application.service;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.supplier.application.dto.SupplierDTO;
import com.metawebthree.supplier.application.dto.SupplierPerformanceDTO;
import com.metawebthree.supplier.application.dto.SupplierRegistrationDTO;
import com.metawebthree.supplier.application.dto.SupplierVerificationDTO;
import com.metawebthree.supplier.domain.model.Supplier;
import com.metawebthree.supplier.domain.repository.SupplierRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class SupplierApplicationService {

    private final SupplierRepository supplierRepository;

    public SupplierApplicationService(SupplierRepository supplierRepository) {
        this.supplierRepository = supplierRepository;
    }

    public SupplierDTO register(SupplierRegistrationDTO registrationDTO) {
        Supplier supplier = new Supplier();
        supplier.setSupplierCode(generateSupplierCode());
        supplier.setSupplierName(registrationDTO.getSupplierName());
        supplier.setContactPerson(registrationDTO.getContactPerson());
        supplier.setContactPhone(registrationDTO.getContactPhone());
        supplier.setContactEmail(registrationDTO.getContactEmail());
        supplier.setAddress(registrationDTO.getAddress());
        supplier.setBusinessLicense(registrationDTO.getBusinessLicense());
        supplier.setLegalPerson(registrationDTO.getLegalPerson());
        supplier.setStatus(Supplier.SupplierStatus.PENDING);
        supplier.setVerificationStatus(Supplier.VerificationStatus.NOT_SUBMITTED);
        supplier.setSupplierLevel(1);
        supplier.setScore(100);
        supplier.setCreateTime(LocalDateTime.now());
        supplier.setUpdateTime(LocalDateTime.now());

        Supplier saved = supplierRepository.save(supplier);
        return toDTO(saved);
    }

    public SupplierDTO submitForVerification(Long supplierId) {
        Supplier supplier = supplierRepository.findById(supplierId)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND));
        supplier.submitForVerification();
        supplier.setUpdateTime(LocalDateTime.now());
        Supplier saved = supplierRepository.save(supplier);
        return toDTO(saved);
    }

    public SupplierDTO verify(SupplierVerificationDTO verificationDTO) {
        Supplier supplier = supplierRepository.findById(verificationDTO.getSupplierId())
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND));
        
        if (Boolean.TRUE.equals(verificationDTO.getApproved())) {
            supplier.approve();
        } else {
            supplier.reject(verificationDTO.getReason());
        }
        supplier.setUpdateTime(LocalDateTime.now());
        Supplier saved = supplierRepository.save(supplier);
        return toDTO(saved);
    }

    public SupplierDTO getSupplierById(Long id) {
        Supplier supplier = supplierRepository.findById(id)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND));
        return toDTO(supplier);
    }

    public SupplierDTO getSupplierByCode(String code) {
        Supplier supplier = supplierRepository.findBySupplierCode(code)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND));
        return toDTO(supplier);
    }

    public List<SupplierDTO> listAllSuppliers() {
        return supplierRepository.findAll().stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }

    public List<SupplierDTO> listSuppliersByStatus(Integer status) {
        return supplierRepository.findByStatus(Supplier.SupplierStatus.fromValue(status)).stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }

    public List<SupplierDTO> listSuppliersByVerificationStatus(Integer verificationStatus) {
        return supplierRepository.findByVerificationStatus(Supplier.VerificationStatus.fromValue(verificationStatus)).stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }

    public SupplierPerformanceDTO evaluatePerformance(Long supplierId) {
        Supplier supplier = supplierRepository.findById(supplierId)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND));
        
        SupplierPerformanceDTO performance = new SupplierPerformanceDTO();
        performance.setSupplierId(supplierId);
        performance.setOrderCount(0);
        performance.setOnTimeDeliveryRate(95);
        performance.setQualityScore(90);
        performance.setResponseTime(85);
        
        int overallScore = (performance.getOnTimeDeliveryRate() + performance.getQualityScore() 
                + performance.getResponseTime()) / 3;
        performance.setOverallScore(overallScore);
        
        supplier.setScore(overallScore);
        supplier.setSupplierLevel(calculateLevel(overallScore));
        supplier.setUpdateTime(LocalDateTime.now());
        supplierRepository.save(supplier);
        
        return performance;
    }

    public SupplierDTO updateScore(Long supplierId, Integer delta) {
        Supplier supplier = supplierRepository.findById(supplierId)
                .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND));
        supplier.updateScore(delta);
        supplier.setSupplierLevel(calculateLevel(supplier.getScore()));
        supplier.setUpdateTime(LocalDateTime.now());
        Supplier saved = supplierRepository.save(supplier);
        return toDTO(saved);
    }

    private Integer calculateLevel(Integer score) {
        if (score >= 90) return 5;
        if (score >= 80) return 4;
        if (score >= 70) return 3;
        if (score >= 60) return 2;
        return 1;
    }

    private String generateSupplierCode() {
        return "SUP-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase();
    }

    private SupplierDTO toDTO(Supplier supplier) {
        SupplierDTO dto = new SupplierDTO();
        dto.setId(supplier.getId());
        dto.setSupplierCode(supplier.getSupplierCode());
        dto.setSupplierName(supplier.getSupplierName());
        dto.setContactPerson(supplier.getContactPerson());
        dto.setContactPhone(supplier.getContactPhone());
        dto.setContactEmail(supplier.getContactEmail());
        dto.setAddress(supplier.getAddress());
        dto.setStatus(supplier.getStatus() != null ? supplier.getStatus().getValue() : null);
        dto.setVerificationStatus(supplier.getVerificationStatus() != null ? supplier.getVerificationStatus().getValue() : null);
        dto.setBusinessLicense(supplier.getBusinessLicense());
        dto.setLegalPerson(supplier.getLegalPerson());
        dto.setSupplierLevel(supplier.getSupplierLevel());
        dto.setScore(supplier.getScore());
        dto.setRemark(supplier.getRemark());
        dto.setCreateTime(supplier.getCreateTime());
        dto.setUpdateTime(supplier.getUpdateTime());
        return dto;
    }
}