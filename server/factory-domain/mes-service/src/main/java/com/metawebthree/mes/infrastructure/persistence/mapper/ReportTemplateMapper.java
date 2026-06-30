package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ReportTemplateDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ReportTemplateMapper extends BaseMapper<ReportTemplateDO> {
}