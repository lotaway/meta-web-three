package com.metawebthree.promotion.application.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeNewProductDO;
import java.util.List;

public interface HomeNewProductService {
    Page<HomeNewProductDO> list(Integer pageNum, Integer pageSize, String productName, Integer recommendStatus);

    void updateRecommendStatus(String ids, Integer recommendStatus);

    void delete(String ids);

    void create(List<HomeNewProductDO> products);

    void updateSort(Long id, Integer sort);
}
