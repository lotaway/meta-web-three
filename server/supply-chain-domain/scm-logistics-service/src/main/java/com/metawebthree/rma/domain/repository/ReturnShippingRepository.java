package com.metawebthree.rma.domain.repository;

import com.metawebthree.rma.domain.entity.ReturnShipping;
import java.util.Optional;

public interface ReturnShippingRepository {
    Optional<ReturnShipping> findByRmaId(Long rmaId);
    Optional<ReturnShipping> findByTrackingNo(String trackingNo);
    ReturnShipping save(ReturnShipping shipping);
}
