package com.metawebthree.inventory.infrastructure.persistence.repository.stockcheck;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckPlan;
import com.metawebthree.inventory.domain.repository.stockcheck.StockCheckPlanRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class StockCheckPlanRepositoryImpl implements StockCheckPlanRepository {

    private final StockCheckPlanMapper mapper;

    @Override
    public StockCheckPlan save(StockCheckPlan plan) {
        if (plan.getId() == null) {
            mapper.insert(plan);
        } else {
            mapper.updateById(plan);
        }
        return plan;
    }

    @Override
    public Optional<StockCheckPlan> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id))
                .filter(p -> !Boolean.TRUE.equals(p.getDeleted()));
    }

    @Override
    public Optional<StockCheckPlan> findByPlanNo(String planNo) {
        return Optional.ofNullable(mapper.selectOne(
                new LambdaQueryWrapper<StockCheckPlan>()
                        .eq(StockCheckPlan::getPlanNo, planNo)
                        .eq(StockCheckPlan::getDeleted, false)));
    }

    @Override
    public List<StockCheckPlan> findAll() {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckPlan>()
                        .eq(StockCheckPlan::getDeleted, false));
    }

    @Override
    public List<StockCheckPlan> findByWarehouseId(Long warehouseId) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckPlan>()
                        .eq(StockCheckPlan::getWarehouseId, warehouseId)
                        .eq(StockCheckPlan::getDeleted, false));
    }

    @Override
    public List<StockCheckPlan> findByStatus(String status) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckPlan>()
                        .eq(StockCheckPlan::getStatus, status)
                        .eq(StockCheckPlan::getDeleted, false));
    }

    @Override
    public List<StockCheckPlan> findByWarehouseIdAndStatus(Long warehouseId, String status) {
        return mapper.selectList(
                new LambdaQueryWrapper<StockCheckPlan>()
                        .eq(StockCheckPlan::getWarehouseId, warehouseId)
                        .eq(StockCheckPlan::getStatus, status)
                        .eq(StockCheckPlan::getDeleted, false));
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
