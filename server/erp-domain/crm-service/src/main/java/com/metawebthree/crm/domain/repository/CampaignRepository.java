package com.metawebthree.crm.domain.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.crm.domain.entity.Campaign;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CampaignRepository extends BaseMapper<Campaign> {
}
