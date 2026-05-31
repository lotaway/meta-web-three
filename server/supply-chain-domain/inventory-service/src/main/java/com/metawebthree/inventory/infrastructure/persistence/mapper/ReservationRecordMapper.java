package com.metawebthree.inventory.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.ReservationRecordDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import java.util.Optional;

@Mapper
public interface ReservationRecordMapper extends BaseMapper<ReservationRecordDO> {
    @Select("SELECT * FROM inventory_reservation_record WHERE biz_id = #{bizId}")
    Optional<ReservationRecordDO> findByBizId(String bizId);
}