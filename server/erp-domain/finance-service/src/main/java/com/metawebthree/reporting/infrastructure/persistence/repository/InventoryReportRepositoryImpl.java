package com.metawebthree.reporting.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.reporting.domain.entity.InventoryReport;
import com.metawebthree.reporting.domain.repository.InventoryReportRepository;
import com.metawebthree.reporting.infrastructure.persistence.converter.InventoryReportConverter;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.InventoryReportDO;
import com.metawebthree.reporting.infrastructure.persistence.mapper.InventoryReportMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class InventoryReportRepositoryImpl implements InventoryReportRepository {

    private final InventoryReportMapper inventoryReportMapper;
    private final InventoryReportConverter inventoryReportConverter;

    public InventoryReportRepositoryImpl(InventoryReportMapper inventoryReportMapper, InventoryReportConverter inventoryReportConverter) {
        this.inventoryReportMapper = inventoryReportMapper;
        this.inventoryReportConverter = inventoryReportConverter;
    }

    @Override
    public Optional<InventoryReport> findById(Long id) {
        InventoryReportDO reportDO = inventoryReportMapper.selectById(id);
        return Optional.ofNullable(inventoryReportConverter.toEntity(reportDO));
    }

    @Override
    public Optional<InventoryReport> findByReportNo(String reportNo) {
        LambdaQueryWrapper<InventoryReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryReportDO::getReportNo, reportNo);
        InventoryReportDO reportDO = inventoryReportMapper.selectOne(wrapper);
        return Optional.ofNullable(inventoryReportConverter.toEntity(reportDO));
    }

    @Override
    public List<InventoryReport> findByType(InventoryReport.ReportType type) {
        LambdaQueryWrapper<InventoryReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryReportDO::getType, type != null ? type.name() : null);
        return inventoryReportMapper.selectList(wrapper).stream()
                .map(inventoryReportConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryReport> findByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end) {
        LambdaQueryWrapper<InventoryReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.ge(InventoryReportDO::getReportDate, start)
               .le(InventoryReportDO::getReportDate, end);
        return inventoryReportMapper.selectList(wrapper).stream()
                .map(inventoryReportConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryReport> findAll() {
        return inventoryReportMapper.selectList(null).stream()
                .map(inventoryReportConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public void save(InventoryReport report) {
        InventoryReportDO reportDO = inventoryReportConverter.toDO(report);
        inventoryReportMapper.insert(reportDO);
    }

    @Override
    public void update(InventoryReport report) {
        InventoryReportDO reportDO = inventoryReportConverter.toDO(report);
        inventoryReportMapper.updateById(reportDO);
    }

    @Override
    public void delete(Long id) {
        inventoryReportMapper.deleteById(id);
    }
}