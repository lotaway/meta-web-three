package com.metawebthree.promotion.application.service;

import com.metawebthree.promotion.infrastructure.persistence.model.CmsPrefrenceAreaDO;
import java.util.List;

public interface CmsPrefrenceAreaService {
    List<CmsPrefrenceAreaDO> listAll();
}
