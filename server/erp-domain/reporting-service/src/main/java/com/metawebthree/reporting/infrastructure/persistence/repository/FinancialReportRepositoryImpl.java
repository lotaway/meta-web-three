package com.metawebthree.reporting.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.reporting.domain.entity.FinancialReport;
import com.metawebthree.reporting.domain.repository.FinancialReportRepository;
import com.metawebthree.reporting.infrastructure.persistence.converter.FinancialReportConverter;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.FinancialReportDO;
import com.metawebthree.reporting.infrastructure.persistence.mapper.FinancialReportMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class FinancialReportRepositoryImpl implements FinancialReportRepository {

    private final FinancialReportMapper financialReportMapper;
    private final FinancialReportConverter financialReportConverter;

    public FinancialReportRepositoryImpl(FinancialReportMapper financialReportMapper, FinancialReportConverter financialReportConverter) {
        this.financialReportMapper = financialReportMapper;
        this.financialReportConverter = financialReportConverter;
    }

    @Override
    public Optional<FinancialReport> findById(Long id) {
        FinancialReportDO reportDO = financialReportMapper.selectById(id);
        return Optional.ofNullable(financialReportConverter.toEntity(reportDO));
    }

    @Override
    public Optional<FinancialReport> findByReportNo(String reportNo) {
        LambdaQueryWrapper<FinancialReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(FinancialReportDO::getReportNo, reportNo);
        FinancialReportDO reportDO = financialReportMapper.selectOne(wrapper);
        return Optional.ofNullable(financialReportConverter.toEntity(reportDO));
    }

    @Override
    public List<FinancialReport> findByType(FinancialReport.ReportType type) {
        LambdaQueryWrapper<FinancialReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(FinancialReportDO::getType, type != null ? type.name() : null);
        return financialReportMapper.selectList(wrapper).stream()
                .map(financialReportConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<FinancialReport> findByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end) {
        LambdaQueryWrapper<FinancialReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.ge(FinancialReportDO::getReportDate, start)
               .le(FinancialReportDO::getReportDate, end);
        return financialReportMapper.selectList(wrapper).stream()
                .map(financialReportConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<FinancialReport> findAll() {
        return financialReportMapper.selectList(null).stream()
                .map(financialReportConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public void save(FinancialReport report) {
        FinancialReportDO reportDO = financialReportConverter.toDO(report);
        financialReportMapper.insert(reportDO);
        report.setId(reportDO.getId());
    }

    @Override
    public void update(FinancialReport report) {
        FinancialReportDO reportDO = financialReportConverter.toDO(report);
        financialReportMapper.updateById(reportDO);
    }

    @Override
    public void delete(Long id) {
        financialReportMapper.deleteById(id);
    }
}