package com.metawebthree.project.infrastructure.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.CostRecord;
import com.metawebthree.project.domain.repository.costRecord.CostRecordRepository;
import com.metawebthree.project.infrastructure.mapper.CostRecordMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.List;

@Repository
@RequiredArgsConstructor
public class CostRecordRepositoryImpl implements CostRecordRepository {

    private final CostRecordMapper costRecordMapper;

    @Override
    public CostRecord save(CostRecord costRecord) {
        costRecordMapper.insert(costRecord);
        return costRecord;
    }

    @Override
    public CostRecord update(CostRecord costRecord) {
        costRecordMapper.updateById(costRecord);
        return costRecord;
    }

    @Override
    public void delete(Long id) {
        costRecordMapper.deleteById(id);
    }

    @Override
    public CostRecord findById(Long id) {
        return costRecordMapper.selectById(id);
    }

    @Override
    public List<CostRecord> findByProjectId(Long projectId) {
        LambdaQueryWrapper<CostRecord> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CostRecord::getProjectId, projectId).orderByDesc(CostRecord::getCostDate);
        return costRecordMapper.selectList(wrapper);
    }

    @Override
    public List<CostRecord> findByDateRange(LocalDate startDate, LocalDate endDate) {
        LambdaQueryWrapper<CostRecord> wrapper = new LambdaQueryWrapper<>();
        wrapper.between(CostRecord::getCostDate, startDate, endDate)
               .orderByDesc(CostRecord::getCostDate);
        return costRecordMapper.selectList(wrapper);
    }

    @Override
    public IPage<CostRecord> findPage(Page<CostRecord> page, Long projectId, String costType, String status, LocalDate startDate, LocalDate endDate) {
        LambdaQueryWrapper<CostRecord> wrapper = new LambdaQueryWrapper<>();
        if (projectId != null) {
            wrapper.eq(CostRecord::getProjectId, projectId);
        }
        if (costType != null && !costType.isEmpty()) {
            wrapper.eq(CostRecord::getCostType, costType);
        }
        if (status != null && !status.isEmpty()) {
            wrapper.eq(CostRecord::getStatus, status);
        }
        if (startDate != null && endDate != null) {
            wrapper.between(CostRecord::getCostDate, startDate, endDate);
        }
        wrapper.orderByDesc(CostRecord::getCostDate);
        return costRecordMapper.selectPage(page, wrapper);
    }
}