package com.metawebthree.promotion.application.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeBrandDO;
import java.util.List;

public interface HomeBrandService {
    Page<HomeBrandDO> list(Integer pageNum, Integer pageSize, String brandName, Integer recommendStatus);

    void updateRecommendStatus(String ids, Integer recommendStatus);

    void delete(String ids);

    void create(List<HomeBrandDO> brands);

    void updateSort(Long id, Integer sort);
}
