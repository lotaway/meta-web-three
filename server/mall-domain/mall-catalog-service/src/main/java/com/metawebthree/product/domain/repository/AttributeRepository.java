package com.metawebthree.product.domain.repository;

import com.metawebthree.product.domain.model.Attribute;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface AttributeRepository {
    void save(Attribute attribute);
    void update(Attribute attribute);
    Attribute findById(Long id);
    List<Attribute> findByCategoryId(Long categoryId);
    void delete(Long id);
}
