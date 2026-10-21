package com.metawebthree.inventory.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.ReservationRecordDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;
import java.util.Optional;

@Mapper
public interface ReservationRecordMapper extends BaseMapper<ReservationRecordDO> {
    @Select("SELECT * FROM inventory_reservation_record WHERE biz_id = #{bizId}")
    Optional<ReservationRecordDO> findByBizId(String bizId);

    @Update("UPDATE inventory_reservation_record SET status = #{nextStatus}, updated_at = NOW() WHERE id = #{id} AND status = #{expectedStatus}")
    int compareAndSetStatus(@Param("id") Long id, @Param("expectedStatus") String expectedStatus, @Param("nextStatus") String nextStatus);
}