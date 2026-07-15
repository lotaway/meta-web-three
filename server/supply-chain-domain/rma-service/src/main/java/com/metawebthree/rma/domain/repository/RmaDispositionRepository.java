package com.metawebthree.rma.domain.repository;

import com.metawebthree.rma.domain.entity.RmaDisposition;
import java.util.Optional;

public interface RmaDispositionRepository {
    Optional<RmaDisposition> findByRmaId(Long rmaId);
    RmaDisposition save(RmaDisposition disposition);
}
