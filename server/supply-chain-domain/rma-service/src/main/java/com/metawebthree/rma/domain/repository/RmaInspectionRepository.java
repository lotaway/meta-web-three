package com.metawebthree.rma.domain.repository;

import com.metawebthree.rma.domain.entity.RmaInspection;
import java.util.Optional;

public interface RmaInspectionRepository {
    Optional<RmaInspection> findByRmaId(Long rmaId);
    RmaInspection save(RmaInspection inspection);
}
