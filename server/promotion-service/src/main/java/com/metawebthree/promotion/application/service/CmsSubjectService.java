package com.metawebthree.promotion.application.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.infrastructure.persistence.model.CmsSubjectDO;
import java.util.List;

public interface CmsSubjectService {
    List<CmsSubjectDO> listAll();
    Page<CmsSubjectDO> list(String keyword, Integer pageNum, Integer pageSize);
    CmsSubjectDO getById(Long id);
    void create(CmsSubjectDO subject);
    void update(Long id, CmsSubjectDO subject);
    void delete(Long id);
}
