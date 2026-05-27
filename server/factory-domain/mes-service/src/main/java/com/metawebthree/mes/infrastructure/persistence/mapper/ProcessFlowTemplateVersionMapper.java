package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessFlowTemplateVersionDO;
import org.apache.ibatis.annotations.Mapper;
import java.util.List;

@Mapper
public interface ProcessFlowTemplateVersionMapper extends BaseMapper<ProcessFlowTemplateVersionDO> {
    
    List<ProcessFlowTemplateVersionDO> selectByTemplateId(Long templateId);
    
    ProcessFlowTemplateVersionDO selectCurrentVersion(Long templateId);
    
    List<ProcessFlowTemplateVersionDO> selectHistoryVersions(Long templateId);
}