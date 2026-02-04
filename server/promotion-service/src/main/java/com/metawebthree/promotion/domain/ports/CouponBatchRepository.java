package com.metawebthree.promotion.domain.ports;

import com.metawebthree.promotion.domain.model.CouponBatch;

public interface CouponBatchRepository {
    void save(CouponBatch batch);
    CouponBatch findById(String id);
}
