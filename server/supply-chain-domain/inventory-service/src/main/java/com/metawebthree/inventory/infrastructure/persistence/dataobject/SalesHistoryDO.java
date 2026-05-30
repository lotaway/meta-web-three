package com.metawebthree.inventory.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("sales_history")
public class SalesHistoryDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private LocalDate salesDate;
    private Integer quantity;
    private String salesChannel;
    private LocalDateTime createdAt;
}