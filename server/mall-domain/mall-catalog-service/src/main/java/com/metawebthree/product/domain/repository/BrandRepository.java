package com.metawebthree.product.domain.repository;

import com.metawebthree.product.domain.model.Brand;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface BrandRepository {
    void save(Brand brand);
    void update(Brand brand);
    Brand findById(Long id);
    List<Brand> findAllBySort();
    void delete(Long id);
}
