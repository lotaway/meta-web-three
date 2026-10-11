package com.metawebthree.user.domain.model;

import com.baomidou.mybatisplus.annotation.EnumValue;
import com.baomidou.mybatisplus.annotation.TableId;
import com.github.yulichang.annotation.Table;
import com.metawebthree.common.utils.UserRole;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Table("User_Role_Mapping")
public class UserRoleMappingDO {
    @TableId
    private Long id;
    // @Column()
    private Long userId;
    @EnumValue()
    private UserRole userRoleId;
}
