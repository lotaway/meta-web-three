package com.metawebthree.order.infrastructure.persistence.mapper;

import java.time.LocalDateTime;
import java.util.List;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.order.domain.model.OutboxEventDO;

@Mapper
public interface OutboxEventMapper extends BaseMapper<OutboxEventDO> {

    @Select("SELECT * FROM tb_outbox_event "
            + "WHERE status = 'PENDING' AND (next_attempt_at IS NULL OR next_attempt_at <= #{now}) "
            + "ORDER BY id ASC LIMIT #{limit}")
    List<OutboxEventDO> findPending(@Param("now") LocalDateTime now, @Param("limit") int limit);

    /**
     * Atomically claim a pending row so that only one relay instance delivers it.
     */
    @Update("UPDATE tb_outbox_event SET status = 'SENDING', updated_at = #{now} "
            + "WHERE id = #{id} AND status = 'PENDING'")
    int claimPending(@Param("id") Long id, @Param("now") LocalDateTime now);

    @Update("UPDATE tb_outbox_event SET status = 'SENT', updated_at = #{now} "
            + "WHERE id = #{id} AND status = 'SENDING'")
    int markSent(@Param("id") Long id, @Param("now") LocalDateTime now);

    @Update("UPDATE tb_outbox_event SET status = #{status}, attempts = attempts + 1, "
            + "last_error = #{error}, next_attempt_at = #{nextAttemptAt}, updated_at = #{now} "
            + "WHERE id = #{id} AND status = 'SENDING'")
    int markAttemptFailed(@Param("id") Long id,
            @Param("status") String status,
            @Param("error") String error,
            @Param("nextAttemptAt") LocalDateTime nextAttemptAt,
            @Param("now") LocalDateTime now);
}
