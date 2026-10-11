package com.metawebthree.supplier.application;

import com.metawebthree.supplier.application.dto.SupplierPerformanceDTO;
import com.metawebthree.supplier.domain.entity.SupplierPerformance;
import com.metawebthree.supplier.domain.repository.SupplierPerformanceRepository;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
public class SupplierPerformanceApplicationService {
    
    private final SupplierPerformanceRepository supplierPerformanceRepository;
    
    public SupplierPerformanceApplicationService(SupplierPerformanceRepository supplierPerformanceRepository) {
        this.supplierPerformanceRepository = supplierPerformanceRepository;
    }
    
    @Transactional
    public SupplierPerformanceDTO createOrUpdateEvaluation(SupplierPerformanceDTO dto) {
        SupplierPerformance performance = toEntity(dto);
        performance.setAssessmentDate(LocalDateTime.now());
        performance.evaluate();
        
        SupplierPerformance existing = supplierPerformanceRepository.findBySupplierIdAndPeriod(
                dto.getSupplierId(), dto.getPeriodStart(), dto.getPeriodEnd());
        
        if (existing != null) {
            performance.setId(existing.getId());
            performance.setCreatedAt(existing.getCreatedAt());
            performance = supplierPerformanceRepository.update(performance);
        } else {
            performance = supplierPerformanceRepository.save(performance);
        }
        
        return toDTO(performance);
    }
    
    public SupplierPerformanceDTO getById(Long id) {
        SupplierPerformance performance = supplierPerformanceRepository.findById(id);
        return performance != null ? toDTO(performance) : null;
    }
    
    public List<SupplierPerformanceDTO> getBySupplierId(Long supplierId) {
        return supplierPerformanceRepository.findBySupplierId(supplierId).stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }
    
    public List<SupplierPerformanceDTO> getAll() {
        return supplierPerformanceRepository.findAll().stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }
    
    public List<SupplierPerformanceDTO> getByAssessmentLevel(String assessmentLevel) {
        return supplierPerformanceRepository.findByAssessmentLevel(assessmentLevel).stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }
    
    @Transactional
    public void deleteById(Long id) {
        supplierPerformanceRepository.deleteById(id);
    }
    
    public Map<String, Object> getDashboard() {
        List<SupplierPerformance> allPerformances = supplierPerformanceRepository.findAll();
        
        if (allPerformances.isEmpty()) {
            return Map.of(
                "totalSuppliers", 0,
                "levelACount", 0,
                "levelBCount", 0,
                "levelCCount", 0,
                "levelDCount", 0,
                "avgOnTimeDeliveryRate", BigDecimal.ZERO,
                "avgQualityPassRate", BigDecimal.ZERO,
                "avgPriceCompetitivenessScore", BigDecimal.ZERO,
                "avgOverallScore", BigDecimal.ZERO,
                "improvementNeededSuppliers", List.of()
            );
        }
        
        long levelACount = allPerformances.stream()
                .filter(p -> "A".equals(p.getAssessmentLevel())).count();
        long levelBCount = allPerformances.stream()
                .filter(p -> "B".equals(p.getAssessmentLevel())).count();
        long levelCCount = allPerformances.stream()
                .filter(p -> "C".equals(p.getAssessmentLevel())).count();
        long levelDCount = allPerformances.stream()
                .filter(p -> "D".equals(p.getAssessmentLevel())).count();
        
        BigDecimal avgOnTimeDeliveryRate = allPerformances.stream()
                .map(SupplierPerformance::getOnTimeDeliveryRate)
                .filter(rate -> rate != null)
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .divide(BigDecimal.valueOf(allPerformances.size()), 2, RoundingMode.HALF_UP);
        
        BigDecimal avgQualityPassRate = allPerformances.stream()
                .map(SupplierPerformance::getQualityPassRate)
                .filter(rate -> rate != null)
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .divide(BigDecimal.valueOf(allPerformances.size()), 2, RoundingMode.HALF_UP);
        
        BigDecimal avgPriceCompetitivenessScore = allPerformances.stream()
                .map(SupplierPerformance::getPriceCompetitivenessScore)
                .filter(score -> score != null)
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .divide(BigDecimal.valueOf(allPerformances.size()), 2, RoundingMode.HALF_UP);
        
        BigDecimal avgOverallScore = allPerformances.stream()
                .map(SupplierPerformance::getOverallScore)
                .filter(score -> score != null)
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .divide(BigDecimal.valueOf(allPerformances.size()), 2, RoundingMode.HALF_UP);
        
        List<SupplierPerformanceDTO> improvementNeeded = allPerformances.stream()
                .filter(p -> "C".equals(p.getAssessmentLevel()) || "D".equals(p.getAssessmentLevel()))
                .sorted((a, b) -> {
                    if (a.getOverallScore() == null) return 1;
                    if (b.getOverallScore() == null) return -1;
                    return a.getOverallScore().compareTo(b.getOverallScore());
                })
                .limit(10)
                .map(this::toDTO)
                .collect(Collectors.toList());
        
        return Map.of(
            "totalSuppliers", (int) allPerformances.stream().map(SupplierPerformance::getSupplierId).distinct().count(),
            "levelACount", (int) levelACount,
            "levelBCount", (int) levelBCount,
            "levelCCount", (int) levelCCount,
            "levelDCount", (int) levelDCount,
            "avgOnTimeDeliveryRate", avgOnTimeDeliveryRate,
            "avgQualityPassRate", avgQualityPassRate,
            "avgPriceCompetitivenessScore", avgPriceCompetitivenessScore,
            "avgOverallScore", avgOverallScore,
            "improvementNeededSuppliers", improvementNeeded
        );
    }
    
    private SupplierPerformanceDTO toDTO(SupplierPerformance entity) {
        if (entity == null) {
            return null;
        }
        SupplierPerformanceDTO dto = new SupplierPerformanceDTO();
        BeanUtils.copyProperties(entity, dto);
        return dto;
    }
    
    private SupplierPerformance toEntity(SupplierPerformanceDTO dto) {
        if (dto == null) {
            return null;
        }
        SupplierPerformance entity = new SupplierPerformance();
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }
}