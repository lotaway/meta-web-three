package com.metawebthree.dom.domain.repository;

import com.metawebthree.dom.domain.entity.DomOrderLine;
import java.util.List;
import java.util.Optional;

public interface DomOrderLineRepository {

    List<DomOrderLine> findByDomOrderId(Long domOrderId);

    List<DomOrderLine> findBySkuCode(String skuCode);

    DomOrderLine save(DomOrderLine line);

    List<DomOrderLine> saveAll(List<DomOrderLine> lines);
}
