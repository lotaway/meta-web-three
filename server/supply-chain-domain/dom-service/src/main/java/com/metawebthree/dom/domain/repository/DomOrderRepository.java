package com.metawebthree.dom.domain.repository;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.dom.domain.entity.DomOrder;
import com.metawebthree.dom.domain.entity.DomOrderStatus;
import java.util.List;
import java.util.Optional;

public interface DomOrderRepository {

    Optional<DomOrder> findById(Long id);

    Optional<DomOrder> findByDomOrderNo(String domOrderNo);

    Optional<DomOrder> findByOriginalOrderNo(String originalOrderNo);

    List<DomOrder> findByStatus(DomOrderStatus status);

    List<DomOrder> findAll();

    IPage<DomOrder> findPage(Page<DomOrder> page, DomOrderStatus status);

    DomOrder save(DomOrder domOrder);
}
