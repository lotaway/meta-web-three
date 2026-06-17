package com.metaweb.datasource.pipeline.repository.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metaweb.datasource.pipeline.repository.entity.LineageNodeDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface LineageNodeMapper extends BaseMapper<LineageNodeDO> {

    @Select("SELECT n.* FROM lineage_node n FINAL " +
            "INNER JOIN lineage_edge e FINAL ON n.node_id = e.source_node_id " +
            "WHERE e.target_node_id = #{nodeId}")
    List<LineageNodeDO> selectUpstreamNodes(@Param("nodeId") String nodeId);

    @Select("SELECT n.* FROM lineage_node n FINAL " +
            "INNER JOIN lineage_edge e FINAL ON n.node_id = e.target_node_id " +
            "WHERE e.source_node_id = #{nodeId}")
    List<LineageNodeDO> selectDownstreamNodes(@Param("nodeId") String nodeId);
}
