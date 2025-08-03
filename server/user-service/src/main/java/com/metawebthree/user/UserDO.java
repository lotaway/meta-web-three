package com.metawebthree.user;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@TableName("User")
public class UserDO {
    private Long id;
    private String username;
    private String nickname;
    private String email;
    private String password;
    private Integer authorId;
    private Short typeId;
    private String walletAddress;
}
