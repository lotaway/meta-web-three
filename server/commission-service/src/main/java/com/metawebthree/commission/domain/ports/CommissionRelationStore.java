package com.metawebthree.commission.domain.ports;

import com.metawebthree.commission.domain.CommissionRelation;

public interface CommissionRelationStore {
    CommissionRelation findByUserId(Long userId);
    void save(CommissionRelation relation);
}
