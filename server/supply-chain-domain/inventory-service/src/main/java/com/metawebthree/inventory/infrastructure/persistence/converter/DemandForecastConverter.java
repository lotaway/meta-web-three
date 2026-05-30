package com.metawebthree.inventory.infrastructure.persistence.converter;

import com.metawebthree.inventory.domain.entity.DemandForecast;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.DemandForecastDO;
import org.springframework.stereotype.Component;
import java.util.ArrayList;
import java.util.List;

@Component
public class DemandForecastConverter {

    public DemandForecast toEntity(DemandForecastDO dto) {
        if (dto == null) {
            return null;
        }
        DemandForecast entity = new DemandForecast();
        entity.setId(dto.getId());
        entity.setSkuCode(dto.getSkuCode());
        entity.setWarehouseId(dto.getWarehouseId());
        entity.setForecastPeriodDays(dto.getForecastPeriodDays());
        entity.setPredictedQuantity(dto.getPredictedQuantity());
        entity.setConfidenceLevel(dto.getConfidenceLevel());
        entity.setForecastMethod(dto.getForecastMethod());
        entity.setForecastStartDate(dto.getForecastStartDate());
        entity.setForecastEndDate(dto.getForecastEndDate());
        entity.setStatus(dto.getStatus());
        entity.setGeneratedAt(dto.getGeneratedAt());
        entity.setNotes(dto.getNotes());
        return entity;
    }

    public DemandForecastDO toDto(DemandForecast entity) {
        if (entity == null) {
            return null;
        }
        DemandForecastDO dto = new DemandForecastDO();
        dto.setId(entity.getId());
        dto.setSkuCode(entity.getSkuCode());
        dto.setWarehouseId(entity.getWarehouseId());
        dto.setForecastPeriodDays(entity.getForecastPeriodDays());
        dto.setPredictedQuantity(entity.getPredictedQuantity());
        dto.setConfidenceLevel(entity.getConfidenceLevel());
        dto.setForecastMethod(entity.getForecastMethod());
        dto.setForecastStartDate(entity.getForecastStartDate());
        dto.setForecastEndDate(entity.getForecastEndDate());
        dto.setStatus(entity.getStatus());
        dto.setGeneratedAt(entity.getGeneratedAt());
        dto.setNotes(entity.getNotes());
        return dto;
    }

    public List<DemandForecast> toEntityList(List<DemandForecastDO> dtoList) {
        if (dtoList == null) {
            return null;
        }
        List<DemandForecast> list = new ArrayList<>();
        for (DemandForecastDO dto : dtoList) {
            list.add(toEntity(dto));
        }
        return list;
    }

    public List<DemandForecastDO> toDtoList(List<DemandForecast> entityList) {
        if (entityList == null) {
            return null;
        }
        List<DemandForecastDO> list = new ArrayList<>();
        for (DemandForecast entity : entityList) {
            list.add(toDto(entity));
        }
        return list;
    }
}