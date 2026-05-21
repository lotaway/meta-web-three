package com.metawebthree.commission.infrastructure.persistence.repository;

import org.springframework.stereotype.Repository;

import com.metawebthree.commission.domain.CommissionConfigData;
import com.metawebthree.commission.domain.ports.CommissionConfigStore;
import com.metawebthree.commission.infrastructure.persistence.mapper.CommissionConfigMapper;
import com.metawebthree.commission.infrastructure.persistence.model.CommissionConfigRecord;

@Repository
public class MybatisCommissionConfigStore implements CommissionConfigStore {
    private final CommissionConfigMapper mapper;

    public MybatisCommissionConfigStore(CommissionConfigMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public CommissionConfigData load() {
        CommissionConfigRecord record = mapper.selectById(1L);
        if (record == null) {
            return null;
        }
        CommissionConfigData data = new CommissionConfigData();
        data.setBuyRate(record.getBuyRate());
        data.setLevelRates(record.getLevelRates());
        data.setMaxLevels(record.getMaxLevels());
        data.setReturnWindowDays(record.getReturnWindowDays());
        return data;
    }
}
