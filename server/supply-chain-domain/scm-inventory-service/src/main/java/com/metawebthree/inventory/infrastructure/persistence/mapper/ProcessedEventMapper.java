package com.metawebthree.inventory.infrastructure.persistence.mapper;

import com.metawebthree.inventory.domain.entity.ProcessedEvent;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface ProcessedEventMapper {

    @Select("SELECT COUNT(1) FROM tb_processed_event "
            + "WHERE event_id = #{eventId} AND consumer_group = #{consumerGroup}")
    int countByEvent(@Param("eventId") String eventId, @Param("consumerGroup") String consumerGroup);

    @Insert("INSERT INTO tb_processed_event (id, event_id, consumer_group, processed_at) "
            + "VALUES (#{id}, #{eventId}, #{consumerGroup}, #{processedAt})")
    int insert(ProcessedEvent event);
}
