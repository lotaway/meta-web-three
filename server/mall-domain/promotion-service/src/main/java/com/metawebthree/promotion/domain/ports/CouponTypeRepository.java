package com.metawebthree.promotion.domain.ports;

import com.metawebthree.promotion.domain.model.CouponType;
import java.time.LocalDateTime;
import java.util.List;

public interface CouponTypeRepository {
    void save(CouponType couponType);
    CouponType findById(Long id);
    List<CouponType> listEnabledActive(LocalDateTime now);
}
