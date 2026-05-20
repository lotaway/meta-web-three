package com.metawebthree.user.domain.model;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@TableName("member_address")
public class MemberAddress extends BaseDO {
    @TableId
    private Long id;
    private Long memberId;
    private String name;
    private String phoneNumber;
    private boolean defaultStatus;
    private String postCode;
    private String province;
    private String city;
    private String region;
    private String detailAddress;
}
