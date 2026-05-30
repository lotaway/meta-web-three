package com.metawebthree.settlement.infrastructure.persistence.repository;

import com.metawebthree.settlement.domain.entity.LogisticsSettlement;
import com.metawebthree.settlement.domain.repository.LogisticsSettlementRepository;
import com.metawebthree.settlement.infrastructure.persistence.converter.LogisticsSettlementConverter;
import com.metawebthree.settlement.infrastructure.persistence.dataobject.LogisticsSettlementDO;
import com.metawebthree.settlement.infrastructure.persistence.mapper.LogisticsSettlementMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class LogisticsSettlementRepositoryImpl implements LogisticsSettlementRepository {
    
    private final LogisticsSettlementMapper mapper;
    private final LogisticsSettlementConverter converter;
    
    public LogisticsSettlementRepositoryImpl(LogisticsSettlementMapper mapper,
                                              LogisticsSettlementConverter converter) {
        this.mapper = mapper;
        this.converter = converter;
    }
    
    @Override
    public LogisticsSettlement save(LogisticsSettlement settlement) {
        LogisticsSettlementDO doObj = converter.toDO(settlement);
        mapper.insert(doObj);
        settlement.setId(doObj.getId());
        return settlement;
    }
    
    @Override
    public Optional<LogisticsSettlement> findById(Long id) {
        LogisticsSettlementDO doObj = mapper.selectById(id);
        return Optional.ofNullable(converter.toEntity(doObj));
    }
    
    @Override
    public Optional<LogisticsSettlement> findByTrackingNo(String trackingNo) {
        LogisticsSettlementDO doObj = mapper.selectOne(
            new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<LogisticsSettlementDO>()
                .eq("tracking_no", trackingNo)
        );
        return Optional.ofNullable(converter.toEntity(doObj));
    }
    
    @Override
    public List<LogisticsSettlement> findByCarrierId(Long carrierId) {
        List<LogisticsSettlementDO> doObjs = mapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<LogisticsSettlementDO>()
                .eq("carrier_id", carrierId)
        );
        return doObjs.stream()
            .map(converter::toEntity)
            .collect(Collectors.toList());
    }
    
    @Override
    public List<LogisticsSettlement> findByStatus(LogisticsSettlement.LogisticsSettlementStatus status) {
        List<LogisticsSettlementDO> doObjs = mapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<LogisticsSettlementDO>()
                .eq("status", status.name())
        );
        return doObjs.stream()
            .map(converter::toEntity)
            .collect(Collectors.toList());
    }
    
    @Override
    public List<LogisticsSettlement> findAll() {
        List<LogisticsSettlementDO> doObjs = mapper.selectList(null);
        return doObjs.stream()
            .map(converter::toEntity)
            .collect(Collectors.toList());
    }
}