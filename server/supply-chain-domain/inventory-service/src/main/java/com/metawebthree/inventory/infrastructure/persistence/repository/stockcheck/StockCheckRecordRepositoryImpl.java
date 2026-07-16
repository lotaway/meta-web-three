package com.metawebthree.inventory.infrastructure.persistence.repository.stockcheck;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckRecord;
import com.metawebthree.inventory.domain.repository.stockcheck.StockCheckRecordRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class StockCheckRecordRepositoryImpl implements StockCheckRecordRepository {

    private final StockCheckRecordMapper mapper;

    @Override
    public StockCheckRecord save(StockCheckRecord record) {
        if (record.getId() == null) {
            mapper.insert(record);
        } else {
            mapper.updateById(record);
        }
        return record;
    }

    @Override
    public Optional<StockCheckRecord> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id))
                .filter(r -> !Boolean.TRUE.equals(r.getDeleted()));
    }

    @Override
    public List<StockCheckRecord> findByPlanId(Long planId) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckRecord>()
                        .eq(StockCheckRecord::getPlanId, planId)
                        .eq(StockCheckRecord::getDeleted, false));
    }

    @Override
    public List<StockCheckRecord> findByPlanNo(String planNo) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckRecord>()
                        .eq(StockCheckRecord::getPlanNo, planNo)
                        .eq(StockCheckRecord::getDeleted, false));
    }

    @Override
    public List<StockCheckRecord> findByWarehouseId(Long warehouseId) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckRecord>()
                        .eq(StockCheckRecord::getWarehouseId, warehouseId)
                        .eq(StockCheckRecord::getDeleted, false));
    }

    @Override
    public List<StockCheckRecord> findByStatus(String status) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckRecord>()
                        .eq(StockCheckRecord::getStatus, status)
                        .eq(StockCheckRecord::getDeleted, false));
    }

    @Override
    public List<StockCheckRecord> findBySkuCode(String skuCode) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckRecord>()
                        .eq(StockCheckRecord::getSkuCode, skuCode)
                        .eq(StockCheckRecord::getDeleted, false));
    }

    @Override
    public List<StockCheckRecord> findHasDifference(Long planId) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckRecord>()
                        .eq(StockCheckRecord::getPlanId, planId)
                        .eq(StockCheckRecord::getDeleted, false)
                        .isNotNull(StockCheckRecord::getDifferenceQuantity)
                        .ne(StockCheckRecord::getDifferenceQuantity, BigDecimal.ZERO));
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
