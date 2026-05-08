package com.metawebthree.order.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("tb_order_return_apply")
public class ReturnApplyDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long orderId;
    private Long companyAddressId;
    private Long productId;
    private String orderSn;
    private LocalDateTime createTime;
    private String memberUsername;
    private BigDecimal returnAmount;
    private String returnName;
    private String returnPhone;
    private Integer status;
    private LocalDateTime handleTime;
    private String productPic;
    private String productName;
    private String productBrand;
    private String productAttr;
    private Integer productCount;
    private BigDecimal productPrice;
    private BigDecimal productRealPrice;
    private String reason;
    private String description;
    private String proofPics;
    private String handleNote;
    private String handleMan;
    private String receiveMan;
    private LocalDateTime receiveTime;
    private String receiveNote;
}
