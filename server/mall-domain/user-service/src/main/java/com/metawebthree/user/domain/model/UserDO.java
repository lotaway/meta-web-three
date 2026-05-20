package com.metawebthree.user.domain.model;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
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
@TableName("User")
public class UserDO extends BaseDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String username;
    private String nickname;
    private String avatar;
    private String email;
    private String password;
    private Long typeId;
}
