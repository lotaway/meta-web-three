package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.DemandForecast;
import com.metawebthree.inventory.infrastructure.persistence.converter.DemandForecastConverter;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.DemandForecastDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.DemandForecastMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;

@Repository
public class DemandForecastRepositoryImpl implements DemandForecastRepository {

    private final DemandForecastMapper mapper;
    private final DemandForecastConverter converter;

    public DemandForecastRepositoryImpl(
            DemandForecastMapper mapper,
            DemandForecastConverter converter) {
        this.mapper = mapper;
        this.converter = converter;
    }

    @Override
    public Optional<DemandForecast> findById(Long id) {
        DemandForecastDO dto = mapper.selectById(id);
        return Optional.ofNullable(converter.toEntity(dto));
    }

    @Override
    public List<DemandForecast> findByStatus(String status) {
        List<DemandForecastDO> dtoList = mapper.selectByStatus(status);
        return converter.toEntityList(dtoList);
    }

    @Override
    public List<DemandForecast> findByWarehouseId(Long warehouseId) {
        List<DemandForecastDO> dtoList = mapper.selectByWarehouseId(warehouseId);
        return converter.toEntityList(dtoList);
    }

    @Override
    public List<DemandForecast> findBySkuAndWarehouse(String skuCode, Long warehouseId) {
        List<DemandForecastDO> dtoList = mapper.selectBySkuAndWarehouse(skuCode, warehouseId);
        return converter.toEntityList(dtoList);
    }

    @Override
    public DemandForecast save(DemandForecast demandForecast) {
        DemandForecastDO dto = converter.toDto(demandForecast);
        if (dto.getId() == null) {
            mapper.insert(dto);
            demandForecast.setId(dto.getId());
        } else {
            mapper.update(dto);
        }
        return demandForecast;
    }

    @Override
    public void delete(DemandForecast demandForecast) {
        if (demandForecast.getId() != null) {
            mapper.deleteById(demandForecast.getId());
        }
    }
}