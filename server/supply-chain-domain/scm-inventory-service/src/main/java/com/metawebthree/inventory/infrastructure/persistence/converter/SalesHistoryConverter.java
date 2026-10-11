package com.metawebthree.inventory.infrastructure.persistence.converter;

import com.metawebthree.inventory.domain.entity.SalesHistory;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.SalesHistoryDO;
import org.springframework.stereotype.Component;
import java.util.ArrayList;
import java.util.List;

@Component
public class SalesHistoryConverter {

    public SalesHistory toEntity(SalesHistoryDO dto) {
        if (dto == null) {
            return null;
        }
        SalesHistory entity = new SalesHistory();
        entity.setId(dto.getId());
        entity.setSkuCode(dto.getSkuCode());
        entity.setWarehouseId(dto.getWarehouseId());
        entity.setSalesDate(dto.getSalesDate());
        entity.setQuantity(dto.getQuantity());
        entity.setSalesChannel(dto.getSalesChannel());
        entity.setCreatedAt(dto.getCreatedAt() != null ? dto.getCreatedAt().toLocalDate() : null);
        return entity;
    }

    public SalesHistoryDO toDto(SalesHistory entity) {
        if (entity == null) {
            return null;
        }
        SalesHistoryDO dto = new SalesHistoryDO();
        dto.setId(entity.getId());
        dto.setSkuCode(entity.getSkuCode());
        dto.setWarehouseId(entity.getWarehouseId());
        dto.setSalesDate(entity.getSalesDate());
        dto.setQuantity(entity.getQuantity());
        dto.setSalesChannel(entity.getSalesChannel());
        dto.setCreatedAt(entity.getCreatedAt() != null ? entity.getCreatedAt().atStartOfDay() : null);
        return dto;
    }

    public List<SalesHistory> toEntityList(List<SalesHistoryDO> dtoList) {
        if (dtoList == null) {
            return null;
        }
        List<SalesHistory> list = new ArrayList<>();
        for (SalesHistoryDO dto : dtoList) {
            list.add(toEntity(dto));
        }
        return list;
    }
}