package com.metawebthree.commission.application;

import java.util.List;

import com.metawebthree.commission.domain.CommissionAccount;
import com.metawebthree.commission.domain.CommissionConfig;
import com.metawebthree.commission.domain.CommissionRecord;
import com.metawebthree.commission.domain.ports.CommissionAccountStore;
import com.metawebthree.commission.domain.ports.CommissionRecordStore;

public class CommissionQueryService {
    private final CommissionRecordStore recordStore;
    private final CommissionAccountStore accountStore;
    private final CommissionConfigProvider configProvider;

    public CommissionQueryService(CommissionRecordStore recordStore,
            CommissionAccountStore accountStore,
            CommissionConfigProvider configProvider) {
        this.recordStore = recordStore;
        this.accountStore = accountStore;
        this.configProvider = configProvider;
    }

    public CommissionAccount getAccount(Long userId) {
        if (userId == null) {
            throw new IllegalArgumentException("invalid user id");
        }
        return accountStore.findByUserId(userId);
    }

    public List<CommissionRecord> listRecords(Long userId, String status, int page, int size) {
        if (userId == null) {
            throw new IllegalArgumentException("invalid user id");
        }
        return recordStore.findByUserId(userId, status, page, size);
    }

    public CommissionConfig getConfig() {
        return configProvider.getConfig();
    }
}
