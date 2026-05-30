package com.metawebthree.finance.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.finance.domain.entity.FinancialRatio;
import com.metawebthree.finance.domain.repository.FinancialRatioRepository;
import com.metawebthree.finance.infrastructure.persistence.converter.FinancialRatioConverter;
import com.metawebthree.finance.infrastructure.persistence.dataobject.FinancialRatioDO;
import com.metawebthree.finance.infrastructure.persistence.mapper.FinancialRatioMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class FinancialRatioRepositoryImpl implements FinancialRatioRepository {

    private final FinancialRatioMapper ratioMapper;
    private final FinancialRatioConverter ratioConverter;

    public FinancialRatioRepositoryImpl(FinancialRatioMapper ratioMapper, FinancialRatioConverter ratioConverter) {
        this.ratioMapper = ratioMapper;
        this.ratioConverter = ratioConverter;
    }

    @Override
    public Optional<FinancialRatio> findById(Long id) {
        FinancialRatioDO ratioDO = ratioMapper.selectById(id);
        return Optional.ofNullable(ratioConverter.toEntity(ratioDO));
    }

    @Override
    public List<FinancialRatio> findByRatioType(String ratioType) {
        LambdaQueryWrapper<FinancialRatioDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(FinancialRatioDO::getRatioType, ratioType);
        List<FinancialRatioDO> ratioDOs = ratioMapper.selectList(wrapper);
        return ratioDOs.stream()
                .map(ratioConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<FinancialRatio> findByPeriod(String period) {
        LambdaQueryWrapper<FinancialRatioDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(FinancialRatioDO::getPeriod, period);
        List<FinancialRatioDO> ratioDOs = ratioMapper.selectList(wrapper);
        return ratioDOs.stream()
                .map(ratioConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<FinancialRatio> findByCalculatedAtBetween(LocalDateTime start, LocalDateTime end) {
        LambdaQueryWrapper<FinancialRatioDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.between(FinancialRatioDO::getCalculatedAt, start, end);
        List<FinancialRatioDO> ratioDOs = ratioMapper.selectList(wrapper);
        return ratioDOs.stream()
                .map(ratioConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<FinancialRatio> findAll() {
        List<FinancialRatioDO> ratioDOs = ratioMapper.selectList(null);
        return ratioDOs.stream()
                .map(ratioConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public void save(FinancialRatio ratio) {
        FinancialRatioDO ratioDO = ratioConverter.toDO(ratio);
        if (ratio.getId() == null) {
            ratioMapper.insert(ratioDO);
            ratio.setId(ratioDO.getId());
        } else {
            ratioMapper.updateById(ratioDO);
        }
    }

    @Override
    public void update(FinancialRatio ratio) {
        FinancialRatioDO ratioDO = ratioConverter.toDO(ratio);
        ratioMapper.updateById(ratioDO);
    }

    @Override
    public void delete(Long id) {
        ratioMapper.deleteById(id);
    }
}