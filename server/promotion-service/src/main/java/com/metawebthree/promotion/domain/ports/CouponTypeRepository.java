package com.metawebthree.promotion.domain.ports;

import com.metawebthree.promotion.domain.model.CouponType;

public interface CouponTypeRepository {
    void save(CouponType couponType);
    CouponType findById(Long id);
}
