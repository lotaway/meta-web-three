package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("sms_home_advertise")
public class AdvertiseRecord {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String name;
    private Integer type;
    private String pic;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Integer status;
    private Integer clickCount;
    private Integer orderCount;
    private String url;
    private String note;
    private Integer sort;
}
