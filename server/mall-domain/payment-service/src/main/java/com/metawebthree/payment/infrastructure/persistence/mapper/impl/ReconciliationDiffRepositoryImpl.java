package com.metawebthree.payment.infrastructure.persistence.mapper.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.payment.domain.repository.ReconciliationDiffRepository;
import com.metawebthree.payment.infrastructure.persistence.dataobject.ReconciliationDiffDO;
import com.metawebthree.payment.infrastructure.persistence.mapper.ReconciliationDiffMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;

@Repository
@RequiredArgsConstructor
public class ReconciliationDiffRepositoryImpl implements ReconciliationDiffRepository {

    private final ReconciliationDiffMapper reconciliationDiffMapper;

    @Override
    public void save(ReconciliationDiffDO diff) {
        reconciliationDiffMapper.insert(diff);
    }

    @Override
    public void saveBatch(List<ReconciliationDiffDO> diffs) {
        for (ReconciliationDiffDO diff : diffs) {
            reconciliationDiffMapper.insert(diff);
        }
    }

    @Override
    public List<ReconciliationDiffDO> findByReconciliationDate(LocalDate reconciliationDate) {
        LambdaQueryWrapper<ReconciliationDiffDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ReconciliationDiffDO::getReconciliationDate, reconciliationDate);
        return reconciliationDiffMapper.selectList(wrapper);
    }

    @Override
    public List<ReconciliationDiffDO> findPendingByReconciliationDate(LocalDate reconciliationDate) {
        LambdaQueryWrapper<ReconciliationDiffDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ReconciliationDiffDO::getReconciliationDate, reconciliationDate)
               .eq(ReconciliationDiffDO::getStatus, ReconciliationDiffDO.DiffStatus.PENDING);
        return reconciliationDiffMapper.selectList(wrapper);
    }

    @Override
    public Long countByReconciliationDateAndDiffType(LocalDate reconciliationDate, String diffType) {
        LambdaQueryWrapper<ReconciliationDiffDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ReconciliationDiffDO::getReconciliationDate, reconciliationDate)
               .eq(ReconciliationDiffDO::getDiffType, diffType);
        return reconciliationDiffMapper.selectCount(wrapper);
    }
}