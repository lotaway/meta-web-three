package com.metawebthree.promotion.application.service.impl;

import com.metawebthree.promotion.application.service.CmsPrefrenceAreaService;
import com.metawebthree.promotion.infrastructure.persistence.mapper.CmsPrefrenceAreaMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.CmsPrefrenceAreaDO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class CmsPrefrenceAreaServiceImpl implements CmsPrefrenceAreaService {
    private final CmsPrefrenceAreaMapper mapper;

    @Override
    public List<CmsPrefrenceAreaDO> listAll() {
        return mapper.selectList(null);
    }
}
