package com.metawebthree.common.DO;

import java.sql.Timestamp;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
public class BaseDO implements Cloneable {
    @TableField(fill = FieldFill.INSERT)
    Timestamp createdAt;
    @TableField(fill = FieldFill.INSERT_UPDATE)
    Timestamp updatedAt;

    @Override
    public BaseDO clone() throws CloneNotSupportedException {
        BaseDO other = new BaseDO();
        other.setCreatedAt(this.getCreatedAt());
        other.setUpdatedAt(this.getUpdatedAt());
        return other;
    }
}
