package com.metawebthree.rma.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.rma.domain.entity.RmaStockInRecord;
import com.metawebthree.rma.domain.repository.RmaStockInRecordRepository;
import com.metawebthree.rma.infrastructure.persistence.converter.RmaStockInRecordConverter;
import com.metawebthree.rma.infrastructure.persistence.dataobject.RmaStockInRecordDO;
import com.metawebthree.rma.infrastructure.persistence.mapper.RmaStockInRecordMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class RmaStockInRecordRepositoryImpl implements RmaStockInRecordRepository {

    private final RmaStockInRecordMapper rmaStockInRecordMapper;
    private final RmaStockInRecordConverter rmaStockInRecordConverter;

    public RmaStockInRecordRepositoryImpl(RmaStockInRecordMapper rmaStockInRecordMapper,
                                          RmaStockInRecordConverter rmaStockInRecordConverter) {
        this.rmaStockInRecordMapper = rmaStockInRecordMapper;
        this.rmaStockInRecordConverter = rmaStockInRecordConverter;
    }

    @Override
    public Optional<RmaStockInRecord> findByRmaIdAndSkuCode(Long rmaId, String skuCode) {
        LambdaQueryWrapper<RmaStockInRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RmaStockInRecordDO::getRmaId, rmaId)
                .eq(RmaStockInRecordDO::getSkuCode, skuCode);
        return rmaStockInRecordMapper.selectList(wrapper).stream()
                .findFirst()
                .map(rmaStockInRecordConverter::toEntity);
    }

    @Override
    public RmaStockInRecord save(RmaStockInRecord record) {
        RmaStockInRecordDO doObj = rmaStockInRecordConverter.toDO(record);
        if (record.getId() == null) {
            rmaStockInRecordMapper.insert(doObj);
            record.setId(doObj.getId());
        } else {
            rmaStockInRecordMapper.updateById(doObj);
        }
        return record;
    }

    @Override
    public List<RmaStockInRecord> findByStatusIn(List<String> statuses) {
        LambdaQueryWrapper<RmaStockInRecordDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.in(RmaStockInRecordDO::getStatus, statuses)
                .orderByAsc(RmaStockInRecordDO::getCreatedAt);
        return rmaStockInRecordMapper.selectList(wrapper).stream()
                .map(rmaStockInRecordConverter::toEntity)
                .collect(Collectors.toList());
    }
}
