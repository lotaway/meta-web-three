package com.metawebthree.inventory.infrastructure.persistence.repository.stockcheck;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckDiff;
import com.metawebthree.inventory.domain.repository.stockcheck.StockCheckDiffRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class StockCheckDiffRepositoryImpl implements StockCheckDiffRepository {

    private final StockCheckDiffMapper mapper;

    @Override
    public StockCheckDiff save(StockCheckDiff diff) {
        if (diff.getId() == null) {
            mapper.insert(diff);
        } else {
            mapper.updateById(diff);
        }
        return diff;
    }

    @Override
    public Optional<StockCheckDiff> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id))
                .filter(d -> !Boolean.TRUE.equals(d.getDeleted()));
    }

    @Override
    public List<StockCheckDiff> findByPlanId(Long planId) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckDiff>()
                        .eq(StockCheckDiff::getPlanId, planId)
                        .eq(StockCheckDiff::getDeleted, false));
    }

    @Override
    public List<StockCheckDiff> findByPlanNo(String planNo) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckDiff>()
                        .eq(StockCheckDiff::getPlanNo, planNo)
                        .eq(StockCheckDiff::getDeleted, false));
    }

    @Override
    public List<StockCheckDiff> findByWarehouseId(Long warehouseId) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckDiff>()
                        .eq(StockCheckDiff::getWarehouseId, warehouseId)
                        .eq(StockCheckDiff::getDeleted, false));
    }

    @Override
    public List<StockCheckDiff> findByProcessingStatus(String status) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckDiff>()
                        .eq(StockCheckDiff::getProcessingStatus, status)
                        .eq(StockCheckDiff::getDeleted, false));
    }

    @Override
    public List<StockCheckDiff> findByApprovalStatus(String status) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckDiff>()
                        .eq(StockCheckDiff::getApprovalStatus, status)
                        .eq(StockCheckDiff::getDeleted, false));
    }

    @Override
    public List<StockCheckDiff> findPendingApproval() {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckDiff>()
                        .eq(StockCheckDiff::getApprovalStatus, StockCheckDiff.APPROVAL_STATUS_PENDING)
                        .eq(StockCheckDiff::getDeleted, false)
                        .apply("ABS(difference_quantity) > {0}", 10));
    }

    @Override
    public List<StockCheckDiff> findBySkuCode(String skuCode) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckDiff>()
                        .eq(StockCheckDiff::getSkuCode, skuCode)
                        .eq(StockCheckDiff::getDeleted, false));
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
