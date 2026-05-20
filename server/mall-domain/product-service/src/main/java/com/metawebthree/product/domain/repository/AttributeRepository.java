package com.metawebthree.product.domain.repository;

import com.metawebthree.product.domain.model.Attribute;
import java.util.List;

public interface AttributeRepository {
    void save(Attribute attribute);
    void update(Attribute attribute);
    Attribute findById(Long id);
    List<Attribute> findByCategoryId(Long categoryId);
    void delete(Long id);
}
