package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_qc_spc_control_chart")
public class SpcControlChartDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String chartCode;
    private String chartName;
    private String chartType;
    private String parameterCode;
    private String limitsJson;
    private String alarmRulesJson;
    private String samplingConfigJson;
    private Boolean isEnabled;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}