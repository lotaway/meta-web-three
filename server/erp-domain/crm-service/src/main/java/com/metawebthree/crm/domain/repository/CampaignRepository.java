package com.metawebthree.crm.domain.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.crm.domain.entity.Campaign;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface CampaignRepository extends BaseMapper<Campaign> {

    @Select("SELECT * FROM crm_campaign WHERE status = #{status} AND deleted = 0")
    List<Campaign> findByStatus(@Param("status") String status);

    @Select("SELECT * FROM crm_campaign WHERE type = #{type} AND deleted = 0")
    List<Campaign> findByType(@Param("type") String type);
}
