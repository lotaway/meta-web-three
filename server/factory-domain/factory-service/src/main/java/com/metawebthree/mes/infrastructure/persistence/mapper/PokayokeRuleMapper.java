package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.PokayokeRuleDO;
import org.apache.ibatis.annotations.Mapper;
import java.util.List;

@Mapper
public interface PokayokeRuleMapper extends BaseMapper<PokayokeRuleDO> {
    
    List<PokayokeRuleDO> selectActiveRules();
    
    List<PokayokeRuleDO> selectByWorkstation(String workstationId);
    
    List<PokayokeRuleDO> selectByProcessCode(String processCode);
}