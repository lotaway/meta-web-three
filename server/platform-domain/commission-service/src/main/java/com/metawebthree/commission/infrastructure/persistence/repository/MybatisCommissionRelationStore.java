package com.metawebthree.commission.infrastructure.persistence.repository;

import org.springframework.stereotype.Repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.commission.domain.CommissionRelation;
import com.metawebthree.commission.domain.ports.CommissionRelationStore;
import com.metawebthree.commission.infrastructure.persistence.mapper.CommissionRelationMapper;
import com.metawebthree.commission.infrastructure.persistence.model.CommissionRelationRecord;

@Repository
public class MybatisCommissionRelationStore implements CommissionRelationStore {
    private final CommissionRelationMapper mapper;

    public MybatisCommissionRelationStore(CommissionRelationMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public CommissionRelation findByUserId(Long userId) {
        CommissionRelationRecord record = mapper.selectOne(new LambdaQueryWrapper<CommissionRelationRecord>()
                .eq(CommissionRelationRecord::getUserId, userId));
        return record == null ? null : toDomain(record);
    }

    @Override
    public void save(CommissionRelation relation) {
        mapper.insert(toRecord(relation));
    }

    private CommissionRelation toDomain(CommissionRelationRecord record) {
        CommissionRelation relation = new CommissionRelation();
        relation.setId(record.getId());
        relation.setUserId(record.getUserId());
        relation.setParentUserId(record.getParentUserId());
        relation.setDepth(record.getDepth());
        relation.setCreatedAt(record.getCreatedAt());
        relation.setUpdatedAt(record.getUpdatedAt());
        return relation;
    }

    private CommissionRelationRecord toRecord(CommissionRelation relation) {
        CommissionRelationRecord record = new CommissionRelationRecord();
        record.setId(relation.getId());
        record.setUserId(relation.getUserId());
        record.setParentUserId(relation.getParentUserId());
        record.setDepth(relation.getDepth());
        record.setCreatedAt(relation.getCreatedAt());
        record.setUpdatedAt(relation.getUpdatedAt());
        return record;
    }
}
