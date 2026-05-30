package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.DemandForecastDTO;
import com.metawebthree.inventory.domain.entity.DemandForecast;
import com.metawebthree.inventory.domain.service.DemandForecastDomainService;
import com.metawebthree.inventory.infrastructure.persistence.repository.DemandForecastRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class DemandForecastApplicationServiceImpl implements DemandForecastApplicationService {

    private final DemandForecastDomainService domainService;
    private final DemandForecastRepository forecastRepository;

    public DemandForecastApplicationServiceImpl(
            DemandForecastDomainService domainService,
            DemandForecastRepository forecastRepository) {
        this.domainService = domainService;
        this.forecastRepository = forecastRepository;
    }

    @Override
    public DemandForecastDTO generateForecast(String skuCode, Long warehouseId, 
            Integer forecastDays, String method) {
        DemandForecast forecast = domainService.generateForecast(skuCode, warehouseId, forecastDays, method);
        return toDTO(forecast);
    }

    @Override
    public List<DemandForecastDTO> generateForecastsForWarehouse(Long warehouseId, 
            Integer forecastDays, String method) {
        List<DemandForecast> forecasts = domainService.generateForecastsForWarehouse(
                warehouseId, forecastDays, method);
        return forecasts.stream().map(this::toDTO).collect(Collectors.toList());
    }

    @Override
    public List<DemandForecastDTO> getPendingForecasts() {
        List<DemandForecast> forecasts = domainService.getPendingForecasts();
        return forecasts.stream().map(this::toDTO).collect(Collectors.toList());
    }

    @Override
    public DemandForecastDTO approveForecast(Long forecastId) {
        DemandForecast forecast = domainService.approveForecast(forecastId);
        return toDTO(forecast);
    }

    @Override
    public DemandForecastDTO rejectForecast(Long forecastId) {
        DemandForecast forecast = domainService.rejectForecast(forecastId);
        return toDTO(forecast);
    }

    @Override
    public DemandForecastDTO queryById(Long id) {
        DemandForecast forecast = forecastRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Demand forecast not found: " + id));
        return toDTO(forecast);
    }

    @Override
    public List<DemandForecastDTO> queryByWarehouse(Long warehouseId) {
        return List.of();
    }

    private DemandForecastDTO toDTO(DemandForecast entity) {
        DemandForecastDTO dto = new DemandForecastDTO();
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
}