package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.AbcClassificationDTO;
import com.metawebthree.inventory.domain.entity.AbcClassification;
import com.metawebthree.inventory.domain.service.InventoryAbcService;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class AbcClassificationApplicationServiceImpl implements AbcClassificationApplicationService {

    private final InventoryAbcService abcService;

    public AbcClassificationApplicationServiceImpl(InventoryAbcService abcService) {
        this.abcService = abcService;
    }

    @Override
    public List<AbcClassificationDTO> classify(Long warehouseId, Integer periodDays) {
        if (periodDays == null || periodDays <= 0) {
            periodDays = 30;
        }
        List<AbcClassification> classifications = abcService.classify(warehouseId, periodDays);
        return classifications.stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }

    private AbcClassificationDTO toDTO(AbcClassification classification) {
        AbcClassificationDTO dto = new AbcClassificationDTO();
        dto.setSkuCode(classification.getSkuCode());
        dto.setCategory(classification.getCategory().name());
        dto.setTotalValue(classification.getTotalValue());
        dto.setTurnoverRate(classification.getTurnoverRate());
        dto.setRank(classification.getRank());
        return dto;
    }
}