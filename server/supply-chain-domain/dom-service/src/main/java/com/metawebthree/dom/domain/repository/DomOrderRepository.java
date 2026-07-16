package com.metawebthree.dom.domain.repository;

import com.metawebthree.dom.domain.entity.DomOrder;
import java.util.List;
import java.util.Optional;

public interface DomOrderRepository {

    Optional<DomOrder> findById(Long id);

    Optional<DomOrder> findByDomOrderNo(String domOrderNo);

    Optional<DomOrder> findByOriginalOrderNo(String originalOrderNo);

    List<DomOrder> findByStatus(String status);

    List<DomOrder> findAll();

    DomOrder save(DomOrder domOrder);
}
