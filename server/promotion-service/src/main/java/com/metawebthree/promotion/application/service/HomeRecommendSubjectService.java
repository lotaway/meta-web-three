package com.metawebthree.promotion.application.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeRecommendSubjectDO;
import java.util.List;

public interface HomeRecommendSubjectService {
    Page<HomeRecommendSubjectDO> list(Integer pageNum, Integer pageSize, String subjectName, Integer recommendStatus);

    void updateRecommendStatus(String ids, Integer recommendStatus);

    void delete(String ids);

    void create(List<HomeRecommendSubjectDO> subjects);

    void updateSort(Long id, Integer sort);
}
