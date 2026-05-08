package com.metawebthree.order.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_company_address")
public class CompanyAddressDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String addressName;
    private Integer sendStatus;
    private Integer receiveStatus;
    private String name;
    private String phone;
    private String province;
    private String city;
    private String region;
    private String detailAddress;
}
