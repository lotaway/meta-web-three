package com.metawebthree.commission.domain.ports;

import com.metawebthree.commission.domain.CommissionConfigData;

public interface CommissionConfigStore {
    CommissionConfigData load();
}
