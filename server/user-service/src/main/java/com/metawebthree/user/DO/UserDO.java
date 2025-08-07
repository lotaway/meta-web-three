package com.metawebthree.user.DO;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("User")
public class UserDO extends BaseDO {
    @TableField(fill = FieldFill.INSERT)
    private Long id;
    private String username;
    private String nickname;
    private String avatar;
    private String email;
    private String password;
}
