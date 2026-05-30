package com.metawebthree.reporting.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.ReportSubscriptionDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface ReportSubscriptionMapper extends BaseMapper<ReportSubscriptionDO> {

    @Select("SELECT * FROM report_subscription WHERE user_id = #{userId}")
    List<ReportSubscriptionDO> selectByUserId(@Param("userId") Long userId);

    @Select("SELECT * FROM report_subscription WHERE enabled = 1")
    List<ReportSubscriptionDO> selectEnabled();

    @Select("SELECT * FROM report_subscription WHERE enabled = 1 AND next_send_time <= #{currentTime}")
    List<ReportSubscriptionDO> selectDueSubscriptions(@Param("currentTime") LocalDateTime currentTime);

    @Select("SELECT * FROM report_subscription WHERE report_type = #{reportType}")
    List<ReportSubscriptionDO> selectByReportType(@Param("reportType") String reportType);
}