package com.metawebthree.user.DO;

import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.utils.UserRole;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@TableName("User_Role")
public class UserRoleDO {
    private Integer id;
    private UserRole role;
}
