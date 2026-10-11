package com.metawebthree.reporting.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.domain.repository.SalesReportRepository;
import com.metawebthree.reporting.infrastructure.persistence.converter.SalesReportConverter;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.SalesReportDO;
import com.metawebthree.reporting.infrastructure.persistence.mapper.SalesReportMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class SalesReportRepositoryImpl implements SalesReportRepository {

    private final SalesReportMapper salesReportMapper;
    private final SalesReportConverter salesReportConverter;

    public SalesReportRepositoryImpl(SalesReportMapper salesReportMapper, SalesReportConverter salesReportConverter) {
        this.salesReportMapper = salesReportMapper;
        this.salesReportConverter = salesReportConverter;
    }

    @Override
    public Optional<SalesReport> findById(Long id) {
        SalesReportDO reportDO = salesReportMapper.selectById(id);
        return Optional.ofNullable(salesReportConverter.toEntity(reportDO));
    }

    @Override
    public Optional<SalesReport> findByReportNo(String reportNo) {
        LambdaQueryWrapper<SalesReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SalesReportDO::getReportNo, reportNo);
        SalesReportDO reportDO = salesReportMapper.selectOne(wrapper);
        return Optional.ofNullable(salesReportConverter.toEntity(reportDO));
    }

    @Override
    public List<SalesReport> findByType(SalesReport.ReportType type) {
        LambdaQueryWrapper<SalesReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SalesReportDO::getType, type != null ? type.name() : null);
        return salesReportMapper.selectList(wrapper).stream()
                .map(salesReportConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<SalesReport> findByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end) {
        LambdaQueryWrapper<SalesReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.ge(SalesReportDO::getReportDate, start)
               .le(SalesReportDO::getReportDate, end);
        return salesReportMapper.selectList(wrapper).stream()
                .map(salesReportConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<SalesReport> findAll() {
        return salesReportMapper.selectList(null).stream()
                .map(salesReportConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public void save(SalesReport report) {
        SalesReportDO reportDO = salesReportConverter.toDO(report);
        salesReportMapper.insert(reportDO);
        report.setId(reportDO.getId());
    }

    @Override
    public void update(SalesReport report) {
        SalesReportDO reportDO = salesReportConverter.toDO(report);
        salesReportMapper.updateById(reportDO);
    }

    @Override
    public void delete(Long id) {
        salesReportMapper.deleteById(id);
    }
}