package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Builder;

import java.time.LocalDateTime;

/**
 * 数据字典�?DO
 */
@Data

@AllArgsConstructor
@NoArgsConstructor
@Builder
@EqualsAndHashCode(callSuper = false)
@TableName("mes_data_dictionary_item")
public class DataDictionaryItemDO {
    
    private Long id;
    private Long dictId;
    private String itemCode;
    private String itemLabel;
    private String parentItemCode;
    private Integer sortOrder;
    private String status; // ACTIVE, INACTIVE
    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;
}