package com.metawebthree.user.domain.model;

import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@TableName("user_profile")
public class UserProfileDO extends BaseDO {
    private Long userId;
    private Integer age;
    private Float externalDebtRatio;
    private Float gpsStability;
    private Integer deviceSharedDegree;
    private String deviceRiskTag;
}
